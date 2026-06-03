"""
Tests for hot-swap knobs (Fase 0).

Verifies that ``LeagueTrainer.set_knob`` / ``set_knobs`` can safely change
key knobs at runtime (without restart) and that the change is picked up by
the relevant code paths (training step, replay buffer, evaluator).

Approach: build a minimal LeagueTrainer instance without running self-play.
We patch out network/training paths so the test is fast and offline.
"""

import os
import sys
import unittest
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


def _make_trainer(tmpdir: str):
    """Build a LeagueTrainer with heavy init paths stubbed out."""
    from train.league.league_trainer import LeagueTrainer
    from train.league.replay_buffer import ReplayBuffer

    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.device = MagicMock()
    trainer.checkpoint_dir = Path(tmpdir) / "ckpt"
    trainer.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainer.log_dir = Path(tmpdir) / "logs"
    trainer.log_dir.mkdir(parents=True, exist_ok=True)
    trainer.use_gpu_batching = False

    # Hot-swap infrastructure
    trainer._state_lock = threading.RLock()
    trainer._pending_changes = {}

    # Adaptive state (set by __init__ normally)
    trainer._num_self_play_workers = 6
    trainer._variant_parallelism = 3
    trainer._buffer_target_size = 100_000
    trainer._last_buffer_target_size = 100_000
    trainer._current_mcts_visits = 200
    trainer.VARIANTS = ["baseline", "attack", "est"]
    trainer.buffers = {
        v: ReplayBuffer(max_size=100_000) for v in trainer.VARIANTS
    }
    trainer.evaluator = MagicMock()
    trainer.evaluator.mcts_visits = 400
    trainer.models = {}
    trainer.optimizers = {}
    trainer.schedulers = {}
    trainer.metrics = MagicMock()
    return trainer


class HotSwapKnobTests(unittest.TestCase):
    """Fase 0: knobs must be changeable at runtime."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def test_set_knob_overrides_class_constant(self):
        self.assertEqual(self.trainer.BATCH_SIZE, 256)  # class default
        ok = self.trainer.set_knob("BATCH_SIZE", 1024)
        self.assertTrue(ok)
        # The instance attribute shadows the class constant
        self.assertEqual(self.trainer.BATCH_SIZE, 1024)

    def test_set_knob_unknown_returns_false(self):
        ok = self.trainer.set_knob("NONEXISTENT", 42)
        self.assertFalse(ok)

    def test_set_knob_rejects_non_settable_knob(self):
        """LR schedule / data loaders are NOT hot-swappable by design."""
        ok = self.trainer.set_knob("INITIAL_LR", 0.01)
        self.assertFalse(ok)
        ok = self.trainer.set_knob("USE_PUZZLE_INJECTION", False)
        self.assertFalse(ok)

    def test_set_knobs_batch(self):
        report = self.trainer.set_knobs({
            "BATCH_SIZE": 512,
            "PUZZLE_BATCHES_PER_GAME_BATCH": 0,
            "PROGAME_BATCHES_PER_GAME_BATCH": 2,
            "POLICY_LOSS_WEIGHT": 2.0,
        })
        self.assertTrue(all(report.values()))
        self.assertEqual(self.trainer.BATCH_SIZE, 512)
        self.assertEqual(self.trainer.PUZZLE_BATCHES_PER_GAME_BATCH, 0)
        self.assertEqual(self.trainer.PROGAME_BATCHES_PER_GAME_BATCH, 2)
        self.assertEqual(self.trainer.POLICY_LOSS_WEIGHT, 2.0)

    def test_immediate_knob_takes_effect_without_round(self):
        """BATCH_SIZE is in the immediate bucket — no _apply_pending_changes needed."""
        self.trainer.set_knob("BATCH_SIZE", 2048)
        self.assertEqual(self.trainer.BATCH_SIZE, 2048)
        self.assertEqual(self.trainer._pending_changes, {})

    def test_deferred_knob_waits_for_apply(self):
        """MCTS_VISITS_SELFPLAY must be applied via _apply_pending_changes."""
        self.trainer.set_knob("MCTS_VISITS_SELFPLAY", 400)
        # Until applied, derived state is unchanged
        self.assertEqual(self.trainer._current_mcts_visits, 200)
        # Apply deferred changes
        self.trainer._apply_pending_changes()
        # Now the derived state is updated
        self.assertEqual(self.trainer._current_mcts_visits, 400)
        self.assertEqual(self.trainer.MCTS_VISITS_SELFPLAY, 400)

    def test_replay_buffer_resize_propagates(self):
        """REPLAY_BUFFER_MAX_SIZE change should resize all variant buffers."""
        for buf in self.trainer.buffers.values():
            self.assertEqual(buf.max_size, 100_000)
        self.trainer.set_knob("REPLAY_BUFFER_MAX_SIZE", 250_000)
        # Immediate? No — this is deferred. Apply now:
        self.trainer._apply_pending_changes()
        for buf in self.trainer.buffers.values():
            self.assertEqual(buf.max_size, 250_000)

    def test_evaluator_mcts_visits_propagates(self):
        """MCTS_VISITS_EVAL change should reach self.evaluator.mcts_visits."""
        self.trainer.set_knob("MCTS_VISITS_EVAL", 800)
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer.evaluator.mcts_visits, 800)

    def test_self_play_workers_propagates(self):
        """NUM_SELF_PLAY_WORKERS change should update _num_self_play_workers."""
        self.trainer.set_knob("NUM_SELF_PLAY_WORKERS", 12)
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer._num_self_play_workers, 12)

    def test_variant_parallelism_propagates(self):
        """SELF_PLAY_VARIANT_PARALLELISM change should update _variant_parallelism."""
        self.trainer.set_knob("SELF_PLAY_VARIANT_PARALLELISM", 2)
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer._variant_parallelism, 2)

    def test_apply_pending_changes_is_idempotent_when_empty(self):
        """No changes queued => no-op, no exceptions."""
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer._current_mcts_visits, 200)  # unchanged

    def test_apply_pending_changes_clears_queue(self):
        self.trainer.set_knob("MCTS_VISITS_SELFPLAY", 500)
        self.assertIn("MCTS_VISITS_SELFPLAY", self.trainer._pending_changes)
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer._pending_changes, {})

    def test_list_hot_knobs_returns_all_known(self):
        snapshot = self.trainer.list_hot_knobs()
        self.assertIn("BATCH_SIZE", snapshot)
        self.assertIn("MCTS_VISITS_SELFPLAY", snapshot)
        self.assertIn("STOCKFISH_BENCH_EVERY_N_ROUNDS", snapshot)
        # Each value is the current effective value
        self.assertEqual(snapshot["BATCH_SIZE"], 256)

    def test_thread_safe_concurrent_set_knob(self):
        """Two threads setting different knobs concurrently must not corrupt state."""
        results = []
        def worker(i):
            self.trainer.set_knob("BATCH_SIZE", 256 + i)
            self.trainer.set_knob("PUZZLE_BATCHES_PER_GAME_BATCH", i % 3)
            results.append((self.trainer.BATCH_SIZE, self.trainer.PUZZLE_BATCHES_PER_GAME_BATCH))
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        # Final value should be one of the values set (256..265)
        self.assertIn(self.trainer.BATCH_SIZE, range(256, 266))
        self.assertIn(self.trainer.PUZZLE_BATCHES_PER_GAME_BATCH, {0, 1, 2})

    def test_no_class_attr_mutation(self):
        """set_knob must not mutate the class constant (only the instance)."""
        from train.league.league_trainer import LeagueTrainer
        original = LeagueTrainer.BATCH_SIZE
        try:
            self.trainer.set_knob("BATCH_SIZE", 4096)
            # Class unchanged
            self.assertEqual(LeagueTrainer.BATCH_SIZE, original)
        finally:
            # Just to be safe, no cleanup needed since we never wrote to class
            pass

    def test_stockfish_knobs_are_hot_settable(self):
        """Stockfish cadence knobs are in the deferred bucket."""
        for knob in ("STOCKFISH_BENCH_EVERY_N_ROUNDS",
                     "STOCKFISH_BENCH_NUM_GAMES",
                     "STOCKFISH_BENCH_TIME_LIMIT_MS"):
            self.assertIn(knob, self.trainer._HOT_KNOBS_DEFERRED)
            self.assertTrue(self.trainer.set_knob(knob, 99))
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer.STOCKFISH_BENCH_EVERY_N_ROUNDS, 99)
        self.assertEqual(self.trainer.STOCKFISH_BENCH_NUM_GAMES, 99)
        self.assertEqual(self.trainer.STOCKFISH_BENCH_TIME_LIMIT_MS, 99)


if __name__ == "__main__":
    unittest.main()
