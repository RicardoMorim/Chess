"""
Spectate mode (Fase 4) - watch model vs model and puzzle drills live.

Two play modes:
  - ``SpectateSession`` (model vs model): both sides use a frozen MCTS
    based on a specified model (variant or specific checkpoint). The game
    is played move-by-move with an ``on_move`` callback invoked after
    each half-move. Cancellation supported via threading.Event.

  - ``PuzzleDrill`` (training drill): model is given a puzzle position
    and tries to find the solution. The drill scores the move as
    correct/incorrect vs the puzzle's expected line. Useful as a
    quick proxy for tactical strength.

``SpectateWorker`` (background thread) drains queued matches from the
trainer's ``_spectate_queue`` and publishes events through the control
server's ``MatchEventBus`` so dashboards can stream them via SSE.
"""

from __future__ import annotations

import logging
import random
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import chess

if False:  # TYPE_CHECKING guard for linter
    from .league_trainer import LeagueTrainer
    from .control_server import MatchEventBus

logger = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================


def _load_model_for_spectate(trainer: "LeagueTrainer", name: str) -> Any:
    """Resolve a model from a name like 'baseline' or 'baseline_step_35'."""
    # If a variant name, use the live in-memory model
    if name in trainer.VARIANTS and name in trainer.models:
        return trainer.models[name]
    # Else, treat as '<variant>_step_<n>' and load the checkpoint
    if "_step_" in name:
        try:
            variant, _, step_str = name.partition("_step_")
            int(step_str)  # validate that it's an int
        except ValueError:
            raise ValueError(f"Cannot parse spectate name: '{name}'")
        ckpt_path = trainer.checkpoint_dir / f"{name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        if variant not in trainer.models:
            raise ValueError(f"Unknown variant: '{variant}'")
        from train.core.models import create_model
        import torch
        # Recreate the model from the saved config
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = state.get("config", {})
        model = create_model(variant=variant, value_dropout=0.0,
                             **{k: v for k, v in cfg.items() if k != "variant"})
        model.load_state_dict(state["state_dict"], strict=False)
        model.to(trainer.device)
        model.eval()
        return model
    raise ValueError(f"Unknown spectate name: '{name}'")


def _san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


# =============================================================================
# Model vs model session
# =============================================================================


@dataclass
class SpectateConfig:
    visits: int = 100
    temperature: float = 0.1
    max_moves: int = 200
    device: Optional[str] = None
    input_channels: Optional[int] = None  # auto-detect from model if None


class SpectateSession:
    """Plays a single game between two frozen MCTSs and emits per-move events."""

    def __init__(
        self,
        trainer: "LeagueTrainer",
        white: Any,
        black: Any,
        start_fen: str = chess.STARTING_FEN,
        config: Optional[SpectateConfig] = None,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.trainer = trainer
        self.white_model = white
        self.black_model = black
        self.start_fen = start_fen
        self.config = config or SpectateConfig(
            visits=trainer.MCTS_VISITS_EVAL,
            device=str(trainer.device),
        )
        if self.config.device is None:
            self.config.device = str(trainer.device)
        self.on_event = on_event or (lambda e: None)
        self._cancel = threading.Event()
        self._autodetect_channels()

    def _autodetect_channels(self) -> None:
        """Derive input channels from white model if not explicitly set."""
        if self.config.input_channels is not None:
            return
        for m in (self.white_model, self.black_model):
            if m is None:
                continue
            try:
                self.config.input_channels = int(m.conv_in.weight.shape[1])
                return
            except Exception:
                continue

    def cancel(self) -> None:
        self._cancel.set()

    def play(self) -> Dict[str, Any]:
        from core.mcts import MCTS
        from core.data import board_to_tensor
        import torch

        cfg = self.config
        device = cfg.device
        in_ch = cfg.input_channels or 18

        common = dict(
            num_visits=cfg.visits,
            temperature=cfg.temperature,
            c_puct=4.0,
            add_noise=False,
        )
        white_mcts = MCTS(model=self.white_model, device=device, **common)
        black_mcts = MCTS(model=self.black_model, device=device, **common)

        board = chess.Board(self.start_fen)
        self.on_event({
            "type": "start",
            "fen": board.fen(),
            "visits": cfg.visits,
            "ply": 0,
        })

        result = "*"
        plies = 0
        for ply in range(cfg.max_moves):
            if self._cancel.is_set():
                result = "*"
                break
            mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            try:
                _, move = mcts.search(board, temperature=cfg.temperature)
            except Exception as e:
                logger.warning(f"MCTS search failed at ply {ply}: {e}")
                break
            if move is None:
                break
            san = _san(board, move)
            board.push(move)
            plies += 1
            # Compute eval (position value from the side-to-move's perspective)
            try:
                inp = torch.tensor(
                    board_to_tensor(board, 0, in_ch),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.white_model(inp)
                eval_white = float(value.item())
            except Exception:
                eval_white = 0.0
            self.on_event({
                "type": "move",
                "fen": board.fen(),
                "move": move.uci(),
                "san": san,
                "ply": plies,
                "eval": eval_white,
            })
            if board.is_game_over():
                result = board.result()
                break

        self.on_event({
            "type": "done",
            "result": result,
            "plies": plies,
            "cancelled": self._cancel.is_set(),
        })
        return {"result": result, "plies": plies, "moves": []}


# =============================================================================
# Puzzle drill
# =============================================================================


@dataclass
class PuzzleSample:
    puzzle_id: str
    fen: str
    solution_moves: List[str]  # uci list
    rating: Optional[int] = None
    themes: List[str] = field(default_factory=list)


class PuzzleDrill:
    """Single-puzzle drill: model tries to find the solution line.

    Each model move is checked against the solution. The drill ends when
    the model makes ``max_wrong`` incorrect moves or runs out of solution.
    """

    def __init__(
        self,
        trainer: "LeagueTrainer",
        model: Any,
        puzzle: PuzzleSample,
        config: Optional[SpectateConfig] = None,
        max_wrong: int = 3,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.trainer = trainer
        self.model = model
        self.puzzle = puzzle
        self.config = config or SpectateConfig(
            visits=trainer.MCTS_VISITS_EVAL,
            device=str(trainer.device),
        )
        if self.config.device is None:
            self.config.device = str(trainer.device)
        self.max_wrong = max_wrong
        self.on_event = on_event or (lambda e: None)
        self._cancel = threading.Event()
        self._autodetect_channels()

    def _autodetect_channels(self) -> None:
        """Derive input channels from the model if not explicitly set."""
        if self.config.input_channels is not None or self.model is None:
            return
        try:
            self.config.input_channels = int(self.model.conv_in.weight.shape[1])
        except Exception:
            pass

    def cancel(self) -> None:
        self._cancel.set()

    def play(self) -> Dict[str, Any]:
        from core.mcts import MCTS
        from core.data import board_to_tensor
        import torch

        cfg = self.config
        device = cfg.device
        in_ch = cfg.input_channels or 18

        mcts = MCTS(
            model=self.model,
            device=device,
            num_visits=cfg.visits,
            temperature=0.0,  # Greedy for drills
            c_puct=4.0,
            add_noise=False,
        )

        board = chess.Board(self.puzzle.fen)
        self.on_event({
            "type": "start",
            "fen": board.fen(),
            "puzzle_id": self.puzzle.puzzle_id,
            "rating": self.puzzle.rating,
            "themes": self.puzzle.themes,
        })

        solution_uci = list(self.puzzle.solution_moves)
        correct = 0
        wrong = 0
        plies = 0
        model_idx = 0  # index in solution_uci of the side-to-move's turn
        while model_idx < len(solution_uci):
            if self._cancel.is_set():
                break
            try:
                _, move = mcts.search(board, temperature=0.0)
            except Exception as e:
                logger.warning(f"MCTS failed during puzzle drill: {e}")
                break
            if move is None:
                break
            expected_uci = solution_uci[model_idx]
            try:
                expected_move = chess.Move.from_uci(expected_uci)
            except Exception:
                logger.warning(f"Bad expected move in puzzle: {expected_uci}")
                break
            is_correct = (move == expected_move)
            san = _san(board, move)
            expected_san = (
                _san(board, expected_move)
                if expected_move in board.legal_moves
                else expected_uci
            )
            try:
                inp = torch.tensor(
                    board_to_tensor(board, 0, in_ch),
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(0)
                with torch.no_grad():
                    _, value = self.model(inp)
                eval_white = float(value.item())
            except Exception:
                eval_white = 0.0
            self.on_event({
                "type": "drill_move",
                "fen": board.fen(),
                "move": move.uci(),
                "san": san,
                "expected_san": expected_san,
                "correct": is_correct,
                "ply": plies + 1,
                "eval": eval_white,
            })
            plies += 1
            if is_correct:
                correct += 1
                board.push(move)
            else:
                wrong += 1
                if wrong >= self.max_wrong:
                    break
                # Push the correct move instead so the puzzle can keep going
                if expected_move in board.legal_moves:
                    board.push(expected_move)
                else:
                    # No legal expected move — puzzle is stuck
                    break
            # Opponent's forced reply
            if model_idx + 1 < len(solution_uci):
                opp_uci = solution_uci[model_idx + 1]
                try:
                    opp_move = chess.Move.from_uci(opp_uci)
                except Exception:
                    break
                if opp_move in board.legal_moves:
                    board.push(opp_move)
                else:
                    # Forced reply isn't actually forced — puzzle ends
                    break
            model_idx += 2

        expected_model_moves = max(
            1, (len(solution_uci) + 1) // 2
        )
        solved = (wrong == 0) and (correct == expected_model_moves)
        self.on_event({
            "type": "done",
            "result": "solved" if solved else "failed",
            "solved": solved,
            "correct": correct,
            "wrong": wrong,
            "plies": plies,
        })
        return {"solved": solved, "correct": correct, "wrong": wrong}


# =============================================================================
# Spectate worker (drains the queue)
# =============================================================================


class SpectateWorker(threading.Thread):
    """Background thread that consumes matches from the trainer's queue
    and runs them, publishing events via the MatchEventBus.

    Stops cleanly when ``stop()`` is called. The trainer can keep
    training in parallel; spectate uses the GPU for forward passes but
    is throttled to one in-flight match at a time.
    """

    def __init__(self, trainer: "LeagueTrainer", match_bus: "MatchEventBus"):
        super().__init__(name="SpectateWorker", daemon=True)
        self.trainer = trainer
        self.match_bus = match_bus
        self._stop = threading.Event()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        # Best-effort wake-up
        try:
            self.trainer._spectate_queue.put_nowait({"_stop": True})
        except Exception:
            pass
        self.join(timeout=timeout)

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                match = self.trainer._spectate_queue.get(timeout=1.0)
            except Exception:
                continue
            if match.get("_stop"):
                break
            try:
                self._run_match(match)
            except Exception as e:
                logger.error(f"Spectate match failed: {e}", exc_info=True)
                self.match_bus.publish({
                    "type": "error",
                    "match_id": match.get("id"),
                    "error": str(e),
                })

    def _run_match(self, match: Dict[str, Any]) -> None:
        params = match.get("params", {})
        match_id = match.get("id")
        mtype = match.get("type")

        def publish(evt: Dict[str, Any]) -> None:
            payload = dict(evt)
            payload["match_id"] = match_id
            self.match_bus.publish(payload)

        if mtype == "model":
            white_name = params.get("white", "baseline")
            black_name = params.get("black", "attack")
            visits = int(params.get("visits", 100))
            start_fen = params.get("start_fen", chess.STARTING_FEN)
            self._run_model_match(white_name, black_name, visits, start_fen, publish)
        elif mtype == "puzzle":
            puzzle_id = params.get("puzzle_id")
            visits = int(params.get("visits", 100))
            self._run_puzzle_drill(puzzle_id, visits, publish)
        else:
            publish({"type": "error", "error": f"unknown match type '{mtype}'"})

    def _run_model_match(self, white_name: str, black_name: str, visits: int,
                          start_fen: str, publish: Callable) -> None:
        try:
            white_model = _load_model_for_spectate(self.trainer, white_name)
            black_model = _load_model_for_spectate(self.trainer, black_name)
        except Exception as e:
            publish({"type": "error", "error": f"model load: {e}"})
            return
        cfg = SpectateConfig(
            visits=visits,
            device=str(self.trainer.device),
            input_channels=int(white_model.conv_in.weight.shape[1]),
        )
        session = SpectateSession(self.trainer, white_model, black_model,
                                  start_fen=start_fen, config=cfg,
                                  on_event=publish)
        session.play()

    def _run_puzzle_drill(self, puzzle_id: Optional[str], visits: int,
                           publish: Callable) -> None:
        # Try to load a puzzle from the cached puzzle tensors or the in-memory
        # puzzle dataset. Fall back to a hard-coded sample.
        puzzle = self._find_puzzle(puzzle_id)
        if puzzle is None:
            publish({"type": "error", "error": "no puzzles available (cache empty?)"})
            return
        # Use baseline model by default
        baseline = self.trainer.models.get("baseline")
        if baseline is None:
            publish({"type": "error", "error": "no baseline model loaded"})
            return
        cfg = SpectateConfig(
            visits=visits,
            device=str(self.trainer.device),
            input_channels=int(baseline.conv_in.weight.shape[1]),
        )
        drill = PuzzleDrill(self.trainer, baseline, puzzle, config=cfg,
                            on_event=publish)
        drill.play()

    def _find_puzzle(self, puzzle_id: Optional[str]) -> Optional[PuzzleSample]:
        """Best-effort puzzle lookup via the ``puzzle_sidecar`` module.

        Loads (or builds) ``train/cache/puzzles_meta.pkl`` and returns
        either a specific puzzle (if ``puzzle_id`` is given and present)
        or a random one. Returns ``None`` if no sidecar can be built.
        """
        # Local import keeps the spectate module lean for non-puzzle use
        from league.puzzle_sidecar import load_puzzle_sidecar

        try:
            puzzles = load_puzzle_sidecar()
        except Exception as e:
            logger.warning(f"puzzle sidecar load failed: {e}")
            puzzles = None
        if not puzzles:
            return None
        if puzzle_id and puzzle_id in puzzles:
            meta = puzzles[puzzle_id]
            chosen_id = puzzle_id
        else:
            # Random pick; seed the caller if determinism is needed
            chosen_id = random.choice(list(puzzles.keys()))
            meta = puzzles[chosen_id]
        return PuzzleSample(
            puzzle_id=chosen_id,
            fen=meta["fen"],
            solution_moves=meta["solution_moves"],
            rating=meta.get("rating"),
            themes=meta.get("themes", []),
        )
