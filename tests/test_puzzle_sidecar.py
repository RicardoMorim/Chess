"""
Tests for the puzzle sidecar builder and the updated PuzzleDrill logic.
"""

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import List

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


SAMPLE_CSV = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\n"
    "TEST1,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,"
    "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,2037,77,95,9125,"
    "crushing hangingPiece long middlegame,https://lichess.org/x,Kings_Pawn\n"
    "TEST2,5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,"
    "d3d6 f8d8 d6d8 f6d8,1455,74,96,35537,advantage endgame short,https://lichess.org/y,\n"
    "BAD1,,,1500,0,0,0,,,\n"  # missing FEN — should be skipped
    "TEST3,8/8/8/8/8/8/4K3/4k3 w - - 0 1,e1d2 e1d2,800,0,0,0,endgame,,\n"  # valid trivial
)


def _write_csv(tmp: str) -> str:
    p = Path(tmp) / "puzzles.csv"
    p.write_text(SAMPLE_CSV, encoding="utf-8")
    return str(p)


def _make_trainer(tmpdir: str):
    from train.league.league_trainer import LeagueTrainer
    from train.league.replay_buffer import ReplayBuffer
    import queue

    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.device = type("D", (), {"__str__": lambda self: "cpu"})()
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
    trainer.VARIANTS = ["baseline"]
    trainer.buffers = {"baseline": ReplayBuffer(max_size=100_000)}
    trainer.evaluator = type("E", (), {"mcts_visits": 400})()
    trainer.models = {}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.metrics = type("M", (), {})()
    trainer.MCTS_VISITS_EVAL = 400
    trainer.performance_mode = "balanced"
    trainer._spectate_queue = queue.Queue()
    return trainer


class PuzzleSidecarBuildTests(unittest.TestCase):

    def test_build_writes_pickle_and_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(tmp)
            out = str(Path(tmp) / "meta.pkl")
            from train.league.puzzle_sidecar import build_puzzle_sidecar
            puzzles = build_puzzle_sidecar(csv_path, output_path=out)
            self.assertEqual(set(puzzles.keys()), {"TEST1", "TEST2", "TEST3"})
            self.assertNotIn("BAD1", puzzles)
            self.assertEqual(puzzles["TEST1"]["fen"],
                "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24")
            self.assertEqual(puzzles["TEST1"]["solution_moves"],
                ["f2g3", "e6e7", "b2b1", "b3c1", "b1c1", "h6c1"])
            self.assertEqual(puzzles["TEST1"]["rating"], 2037)
            self.assertEqual(puzzles["TEST1"]["themes"],
                ["crushing", "hangingPiece", "long", "middlegame"])
            self.assertEqual(puzzles["TEST1"]["opening_tags"], ["Kings_Pawn"])
            self.assertTrue(os.path.exists(out))

    def test_build_with_max_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(tmp)
            out = str(Path(tmp) / "meta.pkl")
            from train.league.puzzle_sidecar import build_puzzle_sidecar
            puzzles = build_puzzle_sidecar(csv_path, output_path=out, max_rows=1)
            self.assertEqual(len(puzzles), 1)
            self.assertIn("TEST1", puzzles)

    def test_build_raises_on_missing_csv(self):
        from train.league.puzzle_sidecar import build_puzzle_sidecar
        with self.assertRaises(FileNotFoundError):
            build_puzzle_sidecar("/no/such/file.csv")

    def test_load_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = _write_csv(tmp)
            out = str(Path(tmp) / "meta.pkl")
            from train.league.puzzle_sidecar import build_puzzle_sidecar, load_puzzle_sidecar
            build_puzzle_sidecar(csv_path, output_path=out)
            puzzles = load_puzzle_sidecar(path=out)
            self.assertIsNotNone(puzzles)
            self.assertEqual(puzzles["TEST2"]["rating"], 1455)

    def test_load_returns_none_when_missing_and_no_autobuild(self):
        from train.league.puzzle_sidecar import load_puzzle_sidecar
        result = load_puzzle_sidecar(
            path="/no/such/file.pkl", auto_build=False
        )
        self.assertIsNone(result)


class PuzzleDrillAlternatingTests(unittest.TestCase):
    """PuzzleDrill plays only on side-to-move turns; opponent replies
    are pushed automatically. Solution list is [model, opp, model, opp, ...].
    The MCTS script fed to the FakeMCTS contains ONLY the side-to-move's
    turns (the opponent's are taken from the solution by the drill).
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def _patch_mcts(self, scripted_moves: List[str]):
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
        fake_module = type(sys)("core.mcts")
        fake_module.MCTS = FakeMCTS
        return unittest.mock.patch.dict(sys.modules, {"core.mcts": fake_module})

    def test_drill_alternates_model_and_opponent(self):
        # FEN is from black's perspective (b - - 0 24) — black is side-to-move.
        # Model script contains only black's turns: f2g3, b2b1, b1c1.
        # Opponent's forced replies (e6e7, b3c1, h6c1) come from the solution.
        model_script = ["f2g3", "b2b1", "b1c1"]
        full_solution = ["f2g3", "e6e7", "b2b1", "b3c1", "b1c1", "h6c1"]
        with self._patch_mcts(model_script):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            events: List[dict] = []
            fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
            drill = PuzzleDrill(
                self.trainer, _fake_model(),
                PuzzleSample(puzzle_id="t1", fen=fen, solution_moves=full_solution),
                on_event=events.append,
            )
            result = drill.play()
        drill_moves = [e for e in events if e["type"] == "drill_move"]
        # Only the side-to-move's turns are drill_move events
        self.assertEqual(len(drill_moves), 3)
        self.assertEqual(drill_moves[0]["move"], "f2g3")
        self.assertTrue(drill_moves[0]["correct"])
        self.assertEqual(drill_moves[1]["move"], "b2b1")
        self.assertTrue(drill_moves[1]["correct"])
        self.assertEqual(drill_moves[2]["move"], "b1c1")
        self.assertTrue(drill_moves[2]["correct"])
        self.assertTrue(result["solved"])

    def test_drill_handles_wrong_first_move(self):
        # Model's only script move is d2d4 (wrong); solution has 2 model turns.
        with self._patch_mcts(["d2d4"]):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
            drill = PuzzleDrill(
                self.trainer, _fake_model(),
                PuzzleSample(puzzle_id="t1", fen=fen,
                             solution_moves=["f2g3", "e6e7", "b2b1", "b3c1"]),
            )
            result = drill.play()
        # First move wrong, then the expected move (f2g3) is auto-played
        # (counted as the model's correct), opponent e6e7 is auto-played,
        # but the next model turn has no script move — puzzle ends.
        self.assertFalse(result["solved"])
        self.assertEqual(result["wrong"], 1)
        self.assertEqual(result["correct"], 0)

    def test_drill_short_circuits_when_max_wrong_hit(self):
        # Model is wrong on every turn; we should stop after max_wrong=2.
        with self._patch_mcts(["d2d4", "e2e4", "f2f4"]):
            from train.league.spectate import PuzzleDrill, PuzzleSample
            fen = "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24"
            drill = PuzzleDrill(
                self.trainer, _fake_model(),
                PuzzleSample(puzzle_id="t1", fen=fen,
                             solution_moves=["f2g3", "e6e7", "b2b1", "b3c1",
                                             "b1c1", "h6c1"],
                             themes=[]),
                max_wrong=2,
            )
            result = drill.play()
        self.assertFalse(result["solved"])
        self.assertEqual(result["wrong"], 2)
        # 2 wrong model turns + auto-pushed correct moves (no correct count)
        self.assertLessEqual(result["wrong"] + result["correct"], 2)


def _fake_model():
    """Lightweight mock with conv_in.weight.shape = [1, 18, 1, 1]."""
    weight = type("W", (), {"shape": [1, 18, 1, 1]})()
    conv = type("C", (), {"weight": weight})()
    return type("M", (), {"conv_in": conv})()


class SpectateFindPuzzleTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)
        self.csv_path = _write_csv(self.tmp)
        self.sidecar_path = str(Path(self.tmp) / "meta.pkl")
        from league.puzzle_sidecar import build_puzzle_sidecar
        build_puzzle_sidecar(self.csv_path, output_path=self.sidecar_path)
        # Import the module via the SAME path spectate uses (league.*).
        # sys.path was set up by the test runner with both '.' and 'train',
        # so 'league.puzzle_sidecar' resolves to the same file but a
        # different sys.modules entry than 'train.league.puzzle_sidecar'.
        import importlib
        self._puzzle_mod = importlib.import_module("league.puzzle_sidecar")
        self._orig_load = self._puzzle_mod.load_puzzle_sidecar
        self._fake_load = self._make_fake_load()
        self._puzzle_mod.load_puzzle_sidecar = self._fake_load

    def tearDown(self):
        self._puzzle_mod.load_puzzle_sidecar = self._orig_load

    def _make_fake_load(self):
        from league.puzzle_sidecar import load_puzzle_sidecar as real
        path = self.sidecar_path

        def fake(*a, **kw):
            return real(path=path, auto_build=False, **kw)
        return fake

    def test_find_by_id(self):
        from train.league.spectate import SpectateWorker
        worker = SpectateWorker.__new__(SpectateWorker)
        worker.trainer = self.trainer
        puzzle = worker._find_puzzle("TEST1")
        self.assertIsNotNone(puzzle)
        self.assertEqual(puzzle.puzzle_id, "TEST1")
        self.assertEqual(puzzle.solution_moves[0], "f2g3")
        self.assertEqual(puzzle.rating, 2037)
        self.assertEqual(puzzle.themes[0], "crushing")

    def test_find_random_when_no_id(self):
        from train.league.spectate import SpectateWorker
        worker = SpectateWorker.__new__(SpectateWorker)
        worker.trainer = self.trainer
        import random
        random.seed(42)
        puzzle = worker._find_puzzle(None)
        self.assertIsNotNone(puzzle)
        self.assertIn(puzzle.puzzle_id, {"TEST1", "TEST2", "TEST3"})

    def test_find_returns_none_when_sidecar_missing(self):
        self._puzzle_mod.load_puzzle_sidecar = lambda *a, **kw: None
        from train.league.spectate import SpectateWorker
        worker = SpectateWorker.__new__(SpectateWorker)
        worker.trainer = self.trainer
        self.assertIsNone(worker._find_puzzle(None))


if __name__ == "__main__":
    unittest.main()
