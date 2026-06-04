"""
Tests for the spectate mode (Fase 4 + Fase 4c Stockfish + mixed players).

Covers:
  - SpectateConfig dataclass defaults
  - PuzzleSample dataclass
  - StockfishConfig / StockfishPlayer lifecycle (engine open/close)
  - MCTSPlayer basic select_move + eval
  - SpectateSession.play with two MCTSPlayer (legacy model-vs-model)
  - SpectateSession with a Stockfish player
  - SpectateSession.play emits events with `side` and `by` fields
  - SpectateSession.cancel interrupts cleanly
  - PuzzleDrill.play scores moves correctly
  - SpectateWorker drains the queue and publishes via MatchEventBus
  - _load_model_for_spectate resolves variant names and checkpoint names
  - _build_player dispatcher (string vs dict, model vs stockfish)
"""

import os
import sys
import threading
import time
import unittest
from typing import List
from unittest.mock import MagicMock, patch
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


class SpectateConfigTests(unittest.TestCase):

    def test_defaults(self):
        from train.league.spectate import SpectateConfig
        c = SpectateConfig()
        self.assertEqual(c.visits, 100)
        self.assertEqual(c.temperature, 0.1)
        self.assertEqual(c.max_moves, 200)

    def test_custom_values(self):
        from train.league.spectate import SpectateConfig
        c = SpectateConfig(visits=400, temperature=0.0, max_moves=50)
        self.assertEqual(c.visits, 400)
        self.assertEqual(c.temperature, 0.0)
        self.assertEqual(c.max_moves, 50)


class PuzzleSampleTests(unittest.TestCase):

    def test_creation(self):
        from train.league.spectate import PuzzleSample
        p = PuzzleSample(
            puzzle_id="abc",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            solution_moves=["e2e4", "e7e5"],
            rating=1500,
            themes=["fork", "pin"],
        )
        self.assertEqual(p.puzzle_id, "abc")
        self.assertEqual(len(p.solution_moves), 2)
        self.assertEqual(p.rating, 1500)
        self.assertEqual(p.themes, ["fork", "pin"])


def _make_trainer(tmpdir: str, with_puzzles: bool = False):
    """Lightweight trainer for spectate tests."""
    from train.league.league_trainer import LeagueTrainer
    from train.league.replay_buffer import ReplayBuffer
    import queue

    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.device = MagicMock()
    trainer.device.__str__ = lambda self: "cpu"
    trainer.checkpoint_dir = Path(tmpdir) / "ckpt"
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.log_dir = Path(tmpdir) / "logs"
    trainer.log_dir.mkdir(parents=True, exist_ok=True)
    trainer.use_gpu_batching = False

    trainer._state_lock = threading.RLock()
    trainer._pending_changes = {}
    trainer._num_self_play_workers = 6
    trainer._variant_parallelism = 3
    trainer._buffer_target_size = 100_000
    trainer._last_buffer_target_size = 100_000
    trainer._current_mcts_visits = 200
    trainer.VARIANTS = ["baseline", "attack", "est"]
    trainer.buffers = {v: ReplayBuffer(max_size=1_000) for v in trainer.VARIANTS}
    trainer.evaluator = MagicMock()
    trainer.evaluator.mcts_visits = 400
    trainer.models = {}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.metrics = MagicMock()
    trainer.MCTS_VISITS_EVAL = 400
    trainer.performance_mode = "balanced"
    trainer._spectate_queue = queue.Queue()
    trainer._training_pause_event = None

    if with_puzzles:
        loader = MagicMock()
        loader._puzzle_ready = lambda: True
        loader.puzzle_dataset = None
        trainer.aux_loader = loader
    else:
        trainer.aux_loader = None
    return trainer


def _patch_mcts_module(scripted_moves: List[str]):
    """Patch ``core.mcts`` so MCTS.search() returns scripted UCI moves.

    Both sides share the same script. Returns a ``patch.dict`` context manager.
    """
    import chess as _chess
    shared = list(scripted_moves)

    class FakeMCTS:
        def __init__(self, model, device, num_visits, temperature, c_puct, add_noise):
            self._moves = shared

        def search(self, board, temperature=None):
            if not self._moves:
                return None, None
            uci = self._moves.pop(0)
            return None, _chess.Move.from_uci(uci)

    fake_mcts_module = MagicMock()
    fake_mcts_module.MCTS = FakeMCTS
    return patch.dict(sys.modules, {"core.mcts": fake_mcts_module})


def _make_mock_model(in_channels: int = 22):
    m = MagicMock()
    m.conv_in.weight.shape = [256, in_channels, 3, 3]
    return m


class MCTSPlayerTests(unittest.TestCase):

    def test_name_and_config(self):
        from train.league.spectate import MCTSPlayer, SpectateConfig
        cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
        p = MCTSPlayer(_make_mock_model(18), "test_model", cfg)
        self.assertEqual(p.name, "test_model")

    def test_eval_returns_value(self):
        from train.league.spectate import MCTSPlayer, SpectateConfig
        import numpy as np
        import torch
        fake_data = MagicMock()
        fake_data.board_to_tensor = lambda b, hist, c: np.zeros((c, 8, 8), dtype=np.float32)
        with patch.dict(sys.modules, {"core.data": fake_data}):
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            class FakeModel:
                class _W:
                    shape = [256, 18, 3, 3]
                conv_in = type("X", (), {"weight": _W()})()
                def __call__(self, x):
                    return torch.zeros(1, 4096), torch.tensor([0.5])
            p = MCTSPlayer(FakeModel(), "baseline", cfg)
            import chess
            v = p.eval(chess.Board())
            self.assertIsNotNone(v, f"eval returned None; check patching")
            self.assertAlmostEqual(v, 0.5, places=3)

    def test_select_move_uses_mcts(self):
        with _patch_mcts_module(["e2e4"]):
            from train.league.spectate import MCTSPlayer, SpectateConfig
            import chess
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            p = MCTSPlayer(_make_mock_model(18), "baseline", cfg)
            move = p.select_move(chess.Board())
            self.assertEqual(move.uci(), "e2e4")


class StockfishConfigTests(unittest.TestCase):

    def test_defaults(self):
        from train.league.spectate import StockfishConfig
        c = StockfishConfig()
        self.assertEqual(c.depth, 12)
        self.assertIsNone(c.time_limit_ms)
        self.assertEqual(c.threads, 1)
        self.assertEqual(c.hash_mb, 64)
        self.assertIsNone(c.path)

    def test_custom_values(self):
        from train.league.spectate import StockfishConfig
        c = StockfishConfig(path="/tmp/sf", depth=20, threads=2, hash_mb=256, time_limit_ms=5000)
        self.assertEqual(c.path, "/tmp/sf")
        self.assertEqual(c.depth, 20)
        self.assertEqual(c.threads, 2)
        self.assertEqual(c.hash_mb, 256)
        self.assertEqual(c.time_limit_ms, 5000)


class StockfishPlayerTests(unittest.TestCase):
    """StockfishPlayer without a real engine — use a fake engine via patch."""

    def test_init_with_path_and_depth(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        cfg = StockfishConfig(path="/tmp/nope/sf", depth=15)
        p = StockfishPlayer(cfg)
        self.assertEqual(p.name, "Stockfish d15")
        p.close()  # no-op without engine

    def test_init_with_time_label(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        cfg = StockfishConfig(path="/tmp/sf", depth=15, time_limit_ms=3000)
        p = StockfishPlayer(cfg)
        self.assertEqual(p.name, "Stockfish t3000ms")

    def test_init_with_custom_label(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        cfg = StockfishConfig(path="/tmp/sf", depth=10)
        p = StockfishPlayer(cfg, label="MyEngine")
        self.assertEqual(p.name, "MyEngine")

    def test_select_move_uses_engine(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        import chess
        cfg = StockfishConfig(path="/tmp/sf", depth=10)
        p = StockfishPlayer(cfg)
        # Inject a fake engine
        fake_engine = MagicMock()
        fake_engine.play.return_value = chess.engine.PlayResult(
            move=chess.Move.from_uci("e2e4"), ponder=None,
            info={}, draw_offered=False, resigned=False,
        )
        p._engine = fake_engine
        move = p.select_move(chess.Board())
        self.assertEqual(move.uci(), "e2e4")
        fake_engine.play.assert_called_once()

    def test_select_move_handles_engine_error(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        import chess
        import chess.engine as ce
        cfg = StockfishConfig(path="/tmp/sf", depth=10)
        p = StockfishPlayer(cfg)
        fake_engine = MagicMock()
        fake_engine.play.side_effect = ce.EngineError("boom")
        p._engine = fake_engine
        move = p.select_move(chess.Board())
        self.assertIsNone(move)

    def test_select_move_without_engine_returns_none(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        import chess
        cfg = StockfishConfig(path="/nonexistent/path/sf.exe", depth=10)
        p = StockfishPlayer(cfg)
        p._failed = True  # simulate failed engine start
        self.assertIsNone(p.select_move(chess.Board()))

    def test_close_quits_engine(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        cfg = StockfishConfig(path="/tmp/sf", depth=10)
        p = StockfishPlayer(cfg)
        fake_engine = MagicMock()
        p._engine = fake_engine
        p.close()
        fake_engine.quit.assert_called_once()
        self.assertIsNone(p._engine)

    def test_close_idempotent(self):
        from train.league.spectate import StockfishConfig, StockfishPlayer
        cfg = StockfishConfig(path="/tmp/sf", depth=10)
        p = StockfishPlayer(cfg)
        p.close()
        p.close()  # second close is safe


class SpectateSessionTests(unittest.TestCase):
    """SpectateSession with two MCTSPlayer (mocked MCTS module)."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def test_session_emits_start_and_done(self):
        with _patch_mcts_module([]):
            from train.league.spectate import MCTSPlayer, SpectateSession, SpectateConfig
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            events: List[dict] = []
            white = MCTSPlayer(_make_mock_model(18), "W", cfg)
            black = MCTSPlayer(_make_mock_model(18), "B", cfg)
            session = SpectateSession(white=white, black=black, config=cfg, on_event=events.append)
            session.play()
        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("done", types)
        start = [e for e in events if e["type"] == "start"][0]
        self.assertEqual(start["white"], "W")
        self.assertEqual(start["black"], "B")
        done = [e for e in events if e["type"] == "done"][0]
        self.assertIn("result", done)
        self.assertIn("plies", done)
        self.assertEqual(done["white"], "W")
        self.assertEqual(done["black"], "B")

    def test_session_emits_move_events_with_side(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
        with _patch_mcts_module(moves):
            from train.league.spectate import MCTSPlayer, SpectateSession, SpectateConfig
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            events: List[dict] = []
            white = MCTSPlayer(_make_mock_model(18), "W", cfg)
            black = MCTSPlayer(_make_mock_model(18), "B", cfg)
            session = SpectateSession(white=white, black=black, config=cfg, max_moves=10, on_event=events.append)
            session.play()
        move_events = [e for e in events if e["type"] == "move"]
        self.assertEqual(move_events[0]["move"], "e2e4")
        self.assertEqual(move_events[0]["san"], "e4")
        self.assertEqual(move_events[0]["ply"], 1)
        self.assertEqual(move_events[0]["side"], "white")
        self.assertEqual(move_events[0]["by"], "W")
        self.assertEqual(move_events[1]["side"], "black")
        self.assertEqual(move_events[1]["by"], "B")
        # Each event has a FEN
        for me in move_events:
            self.assertIn("fen", me)
            self.assertIn("eval", me)

    def test_session_cancel_stops_play(self):
        moves = ["e2e4"] * 100
        with _patch_mcts_module(moves):
            from train.league.spectate import MCTSPlayer, SpectateSession, SpectateConfig
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            events_buf: List[dict] = []
            white = MCTSPlayer(_make_mock_model(18), "W", cfg)
            black = MCTSPlayer(_make_mock_model(18), "B", cfg)
            session = SpectateSession(white=white, black=black, config=cfg, max_moves=200, on_event=events_buf.append)
            orig_append = events_buf.append

            def hook(evt):
                if evt.get("type") == "move":
                    session.cancel()
                orig_append(evt)

            session.on_event = hook
            session.play()
        done = [e for e in events_buf if e["type"] == "done"][0]
        self.assertTrue(done.get("cancelled"))

    def test_session_with_stockfish_player(self):
        """One MCTS player vs one Stockfish player (fake engine)."""
        with _patch_mcts_module(["e2e4"]):
            from train.league.spectate import (
                MCTSPlayer, StockfishConfig, StockfishPlayer,
                SpectateSession, SpectateConfig,
            )
            import chess, chess.engine as ce
            cfg = SpectateConfig(visits=10, device="cpu", input_channels=18)
            white = MCTSPlayer(_make_mock_model(18), "WhiteAI", cfg)

            # Black: Stockfish with a fake engine that picks the first legal move
            sf_cfg = StockfishConfig(path="/tmp/sf", depth=10)
            black = StockfishPlayer(sf_cfg, label="Stockfish")

            def fake_play(board, limit):
                move = next(iter(board.legal_moves), None)
                return ce.PlayResult(
                    move=move, ponder=None, info={},
                    draw_offered=False, resigned=False,
                )

            fake_engine = MagicMock()
            fake_engine.play.side_effect = fake_play
            black._engine = fake_engine  # skip popen

            events: List[dict] = []
            session = SpectateSession(white=white, black=black, config=cfg, max_moves=2, on_event=events.append)
            session.play()

            move_events = [e for e in events if e["type"] == "move"]
            self.assertEqual(len(move_events), 2)
            self.assertEqual(move_events[0]["by"], "WhiteAI")
            self.assertEqual(move_events[0]["move"], "e2e4")
            self.assertEqual(move_events[1]["by"], "Stockfish")
            self.assertEqual(move_events[1]["side"], "black")
            # Stockfish engine was closed in the finally block
            fake_engine.quit.assert_called()


class PuzzleDrillTests(unittest.TestCase):

    def setUp(self):
        import tempfile, chess
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)
        self.fen = chess.STARTING_FEN
        self.solution = ["e2e4", "g8f6", "e4e5", "f6d5"]
        self.model_script = ["e2e4", "e4e5"]

    def _patch_mcts(self, scripted_moves: List[str]):
        return _patch_mcts_module(scripted_moves)

    def test_drill_solves_when_all_correct(self):
        with self._patch_mcts(self.model_script):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            events: List[dict] = []
            drill = PuzzleDrill(
                self.trainer, MagicMock(),
                PuzzleSample(puzzle_id="x", fen=self.fen, solution_moves=self.solution),
                on_event=events.append,
            )
            result = drill.play()
        self.assertTrue(result["solved"])
        self.assertEqual(result["correct"], 2)
        self.assertEqual(result["wrong"], 0)
        move_events = [e for e in events if e["type"] == "drill_move"]
        self.assertEqual(len(move_events), 2)
        self.assertEqual(move_events[0]["move"], "e2e4")
        self.assertEqual(move_events[1]["move"], "e4e5")
        self.assertTrue(all(e["correct"] for e in move_events))

    def test_drill_fails_with_wrong_move(self):
        with self._patch_mcts(["d2d4"]):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            drill = PuzzleDrill(
                self.trainer, MagicMock(),
                PuzzleSample(puzzle_id="x", fen=self.fen, solution_moves=self.solution),
            )
            result = drill.play()
        self.assertFalse(result["solved"])
        self.assertEqual(result["wrong"], 1)

    def test_drill_done_event_has_result(self):
        with self._patch_mcts(self.model_script):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            events: List[dict] = []
            drill = PuzzleDrill(
                self.trainer, MagicMock(),
                PuzzleSample(puzzle_id="x", fen=self.fen, solution_moves=self.solution),
                on_event=events.append,
            )
            drill.play()
        done = [e for e in events if e["type"] == "done"][0]
        self.assertIn("solved", done)
        self.assertEqual(done["result"], "solved")

    def test_drill_move_event_has_side(self):
        """Each drill_move event must include `side` so the dashboard can color it."""
        with self._patch_mcts(self.model_script):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            events: List[dict] = []
            drill = PuzzleDrill(
                self.trainer, MagicMock(),
                PuzzleSample(puzzle_id="x", fen=self.fen, solution_moves=self.solution),
                on_event=events.append,
            )
            drill.play()
        move_events = [e for e in events if e["type"] == "drill_move"]
        self.assertEqual(move_events[0]["side"], "white")
        self.assertEqual(move_events[1]["side"], "white")  # model always white in 1.e4 Nf6 2.e5 Nd5


class SpectateWorkerTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def test_worker_drains_queue_and_publishes(self):
        from train.league.spectate import SpectateWorker
        from train.league.control_server import MatchEventBus
        bus = MatchEventBus()
        worker = SpectateWorker(self.trainer, bus)
        sub = bus.subscribe()
        self.trainer._spectate_queue.put({
            "id": 1,
            "type": "model",
            "params": {"white": "missing_model", "black": "also_missing", "visits": 10},
        })
        worker.start()
        deadline = time.time() + 5
        got = None
        while time.time() < deadline:
            try:
                evt = sub.get(timeout=0.5)
                if evt.get("type") == "error":
                    got = evt
                    break
            except Exception:
                pass
        worker.stop(timeout=2.0)
        self.assertIsNotNone(got, "worker did not publish an error event")
        self.assertIn("missing_model", got.get("error", ""))

    def test_worker_handles_unknown_type(self):
        from train.league.spectate import SpectateWorker
        from train.league.control_server import MatchEventBus
        bus = MatchEventBus()
        worker = SpectateWorker(self.trainer, bus)
        sub = bus.subscribe()
        self.trainer._spectate_queue.put({
            "id": 2, "type": "alien_invasion", "params": {},
        })
        worker.start()
        deadline = time.time() + 5
        got = None
        while time.time() < deadline:
            try:
                evt = sub.get(timeout=0.5)
                if evt.get("type") == "error":
                    got = evt
                    break
            except Exception:
                pass
        worker.stop(timeout=2.0)
        self.assertIsNotNone(got)
        self.assertIn("alien_invasion", got["error"])


class SpectateLoadModelTests(unittest.TestCase):

    def test_load_variant_name(self):
        import tempfile
        from train.league.spectate import _load_model_for_spectate
        tmp = tempfile.mkdtemp()
        trainer = _make_trainer(tmp)
        fake_model = MagicMock()
        trainer.models["baseline"] = fake_model
        loaded = _load_model_for_spectate(trainer, "baseline")
        self.assertIs(loaded, fake_model)

    def test_unknown_name_raises(self):
        import tempfile
        from train.league.spectate import _load_model_for_spectate
        tmp = tempfile.mkdtemp()
        trainer = _make_trainer(tmp)
        with self.assertRaises(ValueError):
            _load_model_for_spectate(trainer, "nonsense")


class BuildPlayerTests(unittest.TestCase):
    """The _build_player dispatcher must produce the right player type."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def _worker(self):
        from train.league.spectate import SpectateWorker
        from train.league.control_server import MatchEventBus
        return SpectateWorker(self.trainer, MatchEventBus())

    def test_string_model_name(self):
        w = self._worker()
        fake_model = _make_mock_model(18)
        self.trainer.models["baseline"] = fake_model
        p = w._build_player("baseline")
        from train.league.spectate import MCTSPlayer
        self.assertIsInstance(p, MCTSPlayer)
        self.assertEqual(p.name, "baseline")

    def test_string_stockfish_shorthand(self):
        from train.league.spectate import StockfishPlayer
        w = self._worker()
        p = w._build_player("stockfish")
        self.assertIsInstance(p, StockfishPlayer)
        # Engine should NOT be started eagerly
        self.assertIsNone(p._engine)
        p.close()

    def test_dict_model(self):
        w = self._worker()
        fake_model = _make_mock_model(18)
        self.trainer.models["attack"] = fake_model
        from train.league.spectate import MCTSPlayer
        p = w._build_player({"type": "model", "name": "attack"})
        self.assertIsInstance(p, MCTSPlayer)
        self.assertEqual(p.name, "attack")

    def test_dict_stockfish(self):
        from train.league.spectate import StockfishPlayer
        w = self._worker()
        p = w._build_player({"type": "stockfish", "depth": 18, "label": "MySF"})
        self.assertIsInstance(p, StockfishPlayer)
        self.assertEqual(p.name, "MySF")
        self.assertEqual(p.config.depth, 18)
        p.close()

    def test_dict_stockfish_with_time(self):
        from train.league.spectate import StockfishPlayer
        w = self._worker()
        p = w._build_player({"type": "stockfish", "time_limit_ms": 5000})
        self.assertIsInstance(p, StockfishPlayer)
        self.assertEqual(p.config.time_limit_ms, 5000)
        p.close()

    def test_unknown_string_raises(self):
        w = self._worker()
        with self.assertRaises(Exception):
            w._build_player("totally_unknown_xyz")

    def test_invalid_descriptor_raises(self):
        w = self._worker()
        with self.assertRaises(ValueError):
            w._build_player(12345)


class BuildModelPlayerTests(unittest.TestCase):

    def test_missing_name_raises(self):
        import tempfile
        from train.league.spectate import SpectateWorker
        from train.league.control_server import MatchEventBus
        tmp = tempfile.mkdtemp()
        trainer = _make_trainer(tmp)
        w = SpectateWorker(trainer, MatchEventBus())
        with self.assertRaises(ValueError):
            w._build_model_player(None)
        with self.assertRaises(ValueError):
            w._build_model_player("")


if __name__ == "__main__":
    import numpy as np
    unittest.main()
