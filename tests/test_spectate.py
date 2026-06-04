"""
Tests for the spectate mode (Fase 4).

Covers:
  - SpectateConfig dataclass defaults
  - PuzzleSample dataclass
  - SpectateSession.play with mocked MCTS publishes move events
  - SpectateSession.cancel interrupts cleanly
  - PuzzleDrill.play scores moves correctly
  - SpectateWorker drains the queue and publishes via MatchEventBus
  - _load_model_for_spectate resolves variant names and checkpoint names
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
    trainer.buffers = {v: ReplayBuffer(max_size=100_000) for v in trainer.VARIANTS}
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

    # Mock the puzzle loader
    if with_puzzles:
        loader = MagicMock()
        loader._puzzle_ready = lambda: True
        loader.puzzle_dataset = None
        trainer.aux_loader = loader
    else:
        trainer.aux_loader = None
    return trainer


class SpectateSessionTests(unittest.TestCase):
    """SpectateSession with a fully-mocked MCTS module."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def _patch_mcts(self, scripted_moves: List[str]):
        """Patch the MCTS class so search() returns scripted moves."""
        import chess as _chess
        from train.league import spectate as sp
        # Build a fake MCTS module
        fake_mcts_module = MagicMock()
        # Shared list so white + black MCTS instances see the same script
        shared = list(scripted_moves)

        class FakeMCTS:
            def __init__(self, model, device, num_visits, temperature, c_puct, add_noise):
                self._moves = shared
            def search(self, board, temperature=None):
                if not self._moves:
                    return None, None
                uci = self._moves.pop(0)
                return None, _chess.Move.from_uci(uci)

        fake_mcts_module.MCTS = FakeMCTS
        return patch.object(sp, "MCTS", FakeMCTS, create=True) if False else patch.dict(
            sys.modules, {"core.mcts": fake_mcts_module}
        )

    def test_session_emits_start_and_done(self):
        with self._patch_mcts([]):
            from train.league.spectate import SpectateSession, SpectateConfig
            events: List[dict] = []
            session = SpectateSession(
                self.trainer, MagicMock(), MagicMock(),
                config=SpectateConfig(visits=10, device="cpu"),
                on_event=events.append,
            )
            session.play()
        types = [e["type"] for e in events]
        self.assertIn("start", types)
        self.assertIn("done", types)
        done = [e for e in events if e["type"] == "done"][0]
        self.assertIn("result", done)
        self.assertIn("plies", done)

    def test_session_emits_move_events(self):
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"]
        with self._patch_mcts(moves):
            from train.league.spectate import SpectateSession, SpectateConfig
            events: List[dict] = []
            session = SpectateSession(
                self.trainer, MagicMock(), MagicMock(),
                config=SpectateConfig(visits=10, device="cpu", max_moves=10),
                on_event=events.append,
            )
            session.play()
        move_events = [e for e in events if e["type"] == "move"]
        # First scripted move is e2e4 (white), then e7e5, etc.
        self.assertEqual(move_events[0]["move"], "e2e4")
        self.assertEqual(move_events[0]["san"], "e4")
        self.assertEqual(move_events[0]["ply"], 1)
        # Each event has a FEN
        for me in move_events:
            self.assertIn("fen", me)
            self.assertIn("eval", me)

    def test_session_cancel_stops_play(self):
        moves = ["e2e4"] * 100  # many scripted moves
        with self._patch_mcts(moves):
            from train.league.spectate import SpectateSession, SpectateConfig
            events: List[dict] = []
            session = SpectateSession(
                self.trainer, MagicMock(), MagicMock(),
                config=SpectateConfig(visits=10, device="cpu", max_moves=200),
                on_event=events.append,
            )
            # Cancel immediately on first move
            events_buf: List[dict] = []
            orig_append = events_buf.append
            def hook(evt):
                if evt.get("type") == "move":
                    session.cancel()
                orig_append(evt)
            session.on_event = hook
            session.play()
        done = [e for e in events_buf if e["type"] == "done"][0]
        self.assertTrue(done.get("cancelled"))

    def test_session_uses_correct_input_channels(self):
        """Input channels come from the model.conv_in.weight shape."""
        with self._patch_mcts([]):
            from train.league.spectate import SpectateSession, SpectateConfig
            # Mock a model with 22 input channels
            mock_model = MagicMock()
            mock_model.conv_in.weight.shape = [256, 22, 3, 3]
            events: List[dict] = []
            session = SpectateSession(
                self.trainer, mock_model, mock_model,
                config=SpectateConfig(visits=10, device="cpu"),
                on_event=events.append,
            )
            session._autodetect_channels()
            self.assertEqual(session.config.input_channels, 22)


class PuzzleDrillTests(unittest.TestCase):
    """PuzzleDrill plays only side-to-move turns; opponent replies are
    pushed from the puzzle's solution_moves. The MCTS script contains
    ONLY the side-to-move's turns.
    """

    def setUp(self):
        import tempfile, chess
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)
        self.fen = chess.STARTING_FEN  # 1.e4 Nf6 2.e5 Nd5
        # Full solution: [model_white, opp_black, model_white, opp_black]
        self.solution = ["e2e4", "g8f6", "e4e5", "f6d5"]
        # Model's script: only white's turns
        self.model_script = ["e2e4", "e4e5"]

    def _patch_mcts(self, scripted_moves: List[str]):
        import chess as _chess
        from train.league import spectate as sp
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
        # Only the side-to-move's turns become drill_move events
        move_events = [e for e in events if e["type"] == "drill_move"]
        self.assertEqual(len(move_events), 2)
        self.assertEqual(move_events[0]["move"], "e2e4")
        self.assertEqual(move_events[1]["move"], "e4e5")
        self.assertTrue(all(e["correct"] for e in move_events))

    def test_drill_fails_with_wrong_move(self):
        with self._patch_mcts(["d2d4"]):  # wrong model move
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
        sub = bus.subscribe()  # subscribe BEFORE putting
        # Enqueue a model match
        self.trainer._spectate_queue.put({
            "id": 1,
            "type": "model",
            "params": {"white": "missing_model", "black": "also_missing", "visits": 10},
        })
        # Start worker, let it process
        worker.start()
        # Wait for error event
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
        sub = bus.subscribe()  # subscribe BEFORE putting
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


if __name__ == "__main__":
    unittest.main()
