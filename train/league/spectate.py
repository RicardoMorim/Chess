"""
Spectate mode (Fase 4) - watch model vs model, model vs Stockfish, and puzzle drills live.

Player types:
  - ``MCTSPlayer`` (frozen MCTS over a model) — used for both variant and
    checkpoint-loaded models.
  - ``StockfishPlayer`` (chess-engine wrapper) — uses an external Stockfish
    binary with depth or time control.
  - The two are interchangeable as long as they expose ``name`` and
    ``select_move(board) -> chess.Move``.

Match kinds:
  - ``model``  - white and black can each be a model or Stockfish. The legacy
    ``white`` + ``black`` fields still work as a shorthand for "both models".
  - ``puzzle`` - model plays a single-puzzle drill. ``PuzzleDrill`` is unchanged.

``SpectateWorker`` (background thread) drains queued matches from the
trainer's ``_spectate_queue`` and publishes events through the control
server's ``MatchEventBus`` so dashboards can stream them via SSE.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

import chess
import chess.engine

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
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = state.get("config", {})
        model = create_model(variant=variant, value_dropout=0.0,
                             **{k: v for k, v in cfg.items() if k != "variant"})
        model.load_state_dict(state["state_dict"], strict=False)
        model.to(trainer.device)
        model.eval()
        return model
    raise ValueError(f"Unknown spectate name: '{name}'")


def _default_stockfish_path() -> str:
    """Locate Stockfish binary, preferring the repo's vendored copy on Windows."""
    # Re-implement here so spectate.py is self-contained (no league.aux_phases import)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidates = [
        os.path.join(repo_root, "stockfish", "stockfish-windows-x86_64-avx2.exe"),
        os.path.join(repo_root, "stockfish", "stockfish"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "stockfish"  # fall back to PATH lookup


def _san(board: chess.Board, move: chess.Move) -> str:
    try:
        return board.san(move)
    except Exception:
        return move.uci()


# =============================================================================
# Player interface
# =============================================================================


class Player(Protocol):
    """Minimal contract: anything that can pick a move for a board position."""

    name: str  # human-readable label for the dashboard

    def select_move(self, board: chess.Board) -> chess.Move: ...


# =============================================================================
# MCTS player (frozen model)
# =============================================================================


@dataclass
class SpectateConfig:
    visits: int = 100
    temperature: float = 0.1
    max_moves: int = 200
    device: Optional[str] = None
    input_channels: Optional[int] = None  # auto-detect from model if None


class MCTSPlayer:
    """Plays moves from a frozen model via MCTS."""

    def __init__(self, model: Any, name: str, config: SpectateConfig):
        self.model = model
        self.name = name
        self.config = config
        self._mcts = None  # lazy

    def _ensure_mcts(self):
        if self._mcts is not None:
            return
        from core.mcts import MCTS
        self._mcts = MCTS(
            model=self.model,
            device=self.config.device,
            num_visits=self.config.visits,
            temperature=self.config.temperature,
            c_puct=4.0,
            add_noise=False,
        )

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        self._ensure_mcts()
        try:
            _, move = self._mcts.search(board, temperature=self.config.temperature)
        except Exception as e:
            logger.warning(f"MCTS search failed in MCTSPlayer '{self.name}': {e}")
            return None
        return move

    def eval(self, board: chess.Board) -> Optional[float]:
        """Return position value from the perspective of the side to move."""
        if self.config.input_channels is None:
            try:
                in_ch = int(self.model.conv_in.weight.shape[1])
            except Exception:
                in_ch = 18
        else:
            in_ch = self.config.input_channels
        try:
            from core.data import board_to_tensor
            import torch
            inp = torch.tensor(
                board_to_tensor(board, 0, in_ch),
                dtype=torch.float32,
                device=self.config.device,
            ).unsqueeze(0)
            with torch.no_grad():
                _, value = self.model(inp)
            return float(value.item())
        except Exception as e:
            logger.debug(f"MCTSPlayer.eval failed: {e!r}")
            return None


# =============================================================================
# Stockfish player
# =============================================================================


@dataclass
class StockfishConfig:
    """Stockfish engine configuration for a single side of a match.

    ``depth`` (plies) and ``time_limit_ms`` are both optional; whichever is
    set takes precedence. Threads default to 1 (each Stockfish instance is
    single-threaded by design; scale by running multiple matches).
    """
    path: Optional[str] = None
    depth: int = 12
    time_limit_ms: Optional[int] = None  # if set, takes precedence over depth
    threads: int = 1
    hash_mb: int = 64


class StockfishPlayer:
    """Plays moves from a running Stockfish engine.

    The engine is started lazily on first move and is closed when
    ``close()`` is called (or at process exit).
    """

    def __init__(self, config: StockfishConfig, label: Optional[str] = None):
        self.config = config
        path = config.path or _default_stockfish_path()
        self.name = label or f"Stockfish d{config.depth}"
        if config.time_limit_ms is not None:
            self.name = label or f"Stockfish t{config.time_limit_ms}ms"
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure_engine(self) -> None:
        if self._engine is not None or self._failed:
            return
        with self._lock:
            if self._engine is not None or self._failed:
                return
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.config.path)
                try:
                    self._engine.configure({
                        "Threads": self.config.threads,
                        "Hash": self.config.hash_mb,
                    })
                except chess.engine.EngineError:
                    pass  # older Stockfish builds may not accept some options
                logger.info(f"StockfishPlayer started: {self.config.path} "
                           f"(threads={self.config.threads}, hash={self.config.hash_mb}MB)")
            except Exception as e:
                logger.error(f"Failed to start Stockfish at '{self.config.path}': {e}")
                self._failed = True

    def select_move(self, board: chess.Board) -> Optional[chess.Move]:
        self._ensure_engine()
        if self._engine is None:
            return None
        if self.config.time_limit_ms is not None:
            limit = chess.engine.Limit(time=self.config.time_limit_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=self.config.depth)
        try:
            result = self._engine.play(board, limit)
            return result.move
        except chess.engine.EngineError as e:
            logger.warning(f"Stockfish play error: {e}")
            return None
        except Exception as e:
            logger.warning(f"Stockfish unexpected error: {e}")
            return None

    def evaluate(self, board: chess.Board) -> Optional[Dict[str, Any]]:
        """Return Stockfish's analysis of the current position.

        Useful for the eval bar in the dashboard. Returns a dict with keys
        ``cp`` (centipawns from white's POV), ``mate`` (mate in N if any),
        and ``depth``.
        """
        self._ensure_engine()
        if self._engine is None:
            return None
        if self.config.time_limit_ms is not None:
            limit = chess.engine.Limit(time=self.config.time_limit_ms / 1000.0)
        else:
            limit = chess.engine.Limit(depth=self.config.depth)
        try:
            with self._engine.analysis(board, limit) as analysis:
                for info in analysis:
                    # Take the first score-bearing info
                    if info.get("score") is not None:
                        score = info["score"].white()
                        return {
                            "cp": score.score(mate_score=10000) if score.score(mate_score=10000) is not None else 0,
                            "mate": score.mate(),
                            "depth": info.get("depth", 0),
                        }
        except Exception as e:
            logger.debug(f"Stockfish evaluate error: {e}")
        return None

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# =============================================================================
# Model vs model session
# =============================================================================


class SpectateSession:
    """Plays a single game between two players (any combination of MCTS / Stockfish)
    and emits per-move events.
    """

    def __init__(
        self,
        white: Player,
        black: Player,
        start_fen: str = chess.STARTING_FEN,
        config: Optional[SpectateConfig] = None,
        max_moves: int = 200,
        on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.white = white
        self.black = black
        self.start_fen = start_fen
        self.config = config or SpectateConfig()
        self.max_moves = max_moves
        self.on_event = on_event or (lambda e: None)
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def play(self) -> Dict[str, Any]:
        board = chess.Board(self.start_fen)
        self.on_event({
            "type": "start",
            "fen": board.fen(),
            "visits": self.config.visits,
            "ply": 0,
            "white": self.white.name,
            "black": self.black.name,
        })

        result = "*"
        plies = 0
        try:
            for ply in range(self.max_moves):
                if self._cancel.is_set():
                    result = "*"
                    break
                current = self.white if board.turn == chess.WHITE else self.black
                move = current.select_move(board)
                if move is None:
                    logger.warning(f"Player '{current.name}' returned no move at ply {ply}")
                    break
                san = _san(board, move)
                side = "white" if board.turn == chess.WHITE else "black"
                board.push(move)
                plies += 1
                # Compute eval from the MCTS player's perspective (or fall back
                # to a cp-only Stockfish eval). Both go on the same wire format
                # so the dashboard's eval bar works unchanged.
                eval_value = self._eval_position(board)
                self.on_event({
                    "type": "move",
                    "fen": board.fen(),
                    "move": move.uci(),
                    "san": san,
                    "ply": plies,
                    "side": side,
                    "by": current.name,
                    "eval": eval_value,
                })
                if board.is_game_over():
                    result = board.result()
                    break
        finally:
            # Best-effort cleanup of Stockfish subprocesses
            for p in (self.white, self.black):
                if isinstance(p, StockfishPlayer):
                    try:
                        p.close()
                    except Exception:
                        pass

        self.on_event({
            "type": "done",
            "result": result,
            "plies": plies,
            "cancelled": self._cancel.is_set(),
            "white": self.white.name,
            "black": self.black.name,
        })
        return {"result": result, "plies": plies, "moves": []}

    def _eval_position(self, board: chess.Board) -> Optional[float]:
        """Get a [-1, 1] evaluation. Prefer MCTS player model; fall back to
        Stockfish analysis from the available Stockfish player (if any).
        """
        # Prefer the MCTS player (model-based, fast)
        for p in (self.white, self.black):
            if isinstance(p, MCTSPlayer):
                v = p.eval(board)
                if v is not None:
                    return v
        # Fall back to a Stockfish eval
        for p in (self.white, self.black):
            if isinstance(p, StockfishPlayer):
                info = p.evaluate(board)
                if info is None:
                    continue
                cp = info.get("cp")
                if cp is None:
                    return 0.0
                # Centipawns to [-1, 1] value (sigmoid with 400 cp scale, like AlphaZero)
                return max(-1.0, min(1.0, cp / 400.0))
        return None


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
                "side": "white" if board.turn == chess.WHITE else "black",
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
                if expected_move in board.legal_moves:
                    board.push(expected_move)
                else:
                    break
            if model_idx + 1 < len(solution_uci):
                opp_uci = solution_uci[model_idx + 1]
                try:
                    opp_move = chess.Move.from_uci(opp_uci)
                except Exception:
                    break
                if opp_move in board.legal_moves:
                    board.push(opp_move)
                else:
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

    # ---- dispatch ---------------------------------------------------------

    def _run_match(self, match: Dict[str, Any]) -> None:
        params = match.get("params", {})
        match_id = match.get("id")
        mtype = match.get("type")

        def publish(evt: Dict[str, Any]) -> None:
            payload = dict(evt)
            payload["match_id"] = match_id
            self.match_bus.publish(payload)

        if mtype == "model":
            self._run_model_match(params, publish)
        elif mtype == "puzzle":
            puzzle_id = params.get("puzzle_id")
            visits = int(params.get("visits", 100))
            self._run_puzzle_drill(puzzle_id, visits, publish)
        else:
            publish({"type": "error", "error": f"unknown match type '{mtype}'"})

    # ---- builders ---------------------------------------------------------

    def _build_model_player(self, descriptor: Any) -> MCTSPlayer:
        """Build an MCTSPlayer from a descriptor like 'baseline',
        'baseline_step_35', or {'type':'model', 'name':'baseline_step_35'}.

        Raises on error (caller catches and reports via the match bus).
        """
        if isinstance(descriptor, dict):
            name = descriptor.get("name")
        else:
            name = descriptor
        if not name:
            raise ValueError("model descriptor missing 'name'")
        model = _load_model_for_spectate(self.trainer, name)
        try:
            in_ch = int(model.conv_in.weight.shape[1])
        except Exception:
            in_ch = 18
        cfg = SpectateConfig(
            visits=0,  # overwritten by caller's value below
            device=str(self.trainer.device),
            input_channels=in_ch,
            temperature=0.1,
        )
        return MCTSPlayer(model, name, cfg)

    def _build_stockfish_player(self, descriptor: Any) -> Optional[StockfishPlayer]:
        if not isinstance(descriptor, dict):
            return None
        if descriptor.get("type") != "stockfish":
            return None
        cfg = StockfishConfig(
            path=descriptor.get("path"),
            depth=int(descriptor.get("depth", 12)),
            time_limit_ms=descriptor.get("time_limit_ms"),
            threads=int(descriptor.get("threads", 1)),
            hash_mb=int(descriptor.get("hash_mb", 64)),
        )
        label = descriptor.get("label") or (
            f"Stockfish d{cfg.depth}" if cfg.time_limit_ms is None
            else f"Stockfish t{cfg.time_limit_ms}ms"
        )
        return StockfishPlayer(cfg, label=label)

    def _build_player(self, descriptor: Any, default_engine: str = "model") -> Player:
        """Build a player from a shorthand string OR a dict.

        Shorthand:
          "baseline"                       -> model variant
          "baseline_step_35"               -> model checkpoint
          "stockfish"                      -> Stockfish with default depth

        Dict form (richer control):
          {"type": "model", "name": "..."}         -> MCTSPlayer
          {"type": "stockfish", "depth": 12, ...}  -> StockfishPlayer
        """
        if isinstance(descriptor, str):
            if descriptor.lower() in ("stockfish", "sf"):
                return self._build_stockfish_player({"type": "stockfish"})
            return self._build_model_player(descriptor)
        if isinstance(descriptor, dict):
            t = descriptor.get("type", default_engine)
            if t == "stockfish":
                return self._build_stockfish_player(descriptor)
            return self._build_model_player(descriptor)
        raise ValueError(f"invalid player descriptor: {descriptor!r}")

    # ---- runners ----------------------------------------------------------

    def _run_model_match(self, params: Dict[str, Any], publish: Callable) -> None:
        visits = int(params.get("visits", 100))
        start_fen = params.get("start_fen", chess.STARTING_FEN)
        # White / black are flexible: string shorthand or {type, name} dict
        try:
            white = self._build_player(params.get("white", "baseline"))
            black = self._build_player(params.get("black", "attack"))
        except Exception as e:
            publish({"type": "error", "error": f"player build: {e}"})
            return
        # Push the visit count into the MCTS configs (if they have one)
        for p in (white, black):
            if isinstance(p, MCTSPlayer):
                p.config.visits = visits
        session = SpectateSession(
            white=white, black=black,
            start_fen=start_fen, max_moves=int(params.get("max_moves", 200)),
            on_event=publish,
        )
        session.play()

    def _run_puzzle_drill(self, puzzle_id: Optional[str], visits: int,
                           publish: Callable) -> None:
        puzzle = self._find_puzzle(puzzle_id)
        if puzzle is None:
            publish({"type": "error", "error": "no puzzles available (cache empty?)"})
            return
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
            chosen_id = random.choice(list(puzzles.keys()))
            meta = puzzles[chosen_id]
        return PuzzleSample(
            puzzle_id=chosen_id,
            fen=meta["fen"],
            solution_moves=meta["solution_moves"],
            rating=meta.get("rating"),
            themes=meta.get("themes", []),
        )
