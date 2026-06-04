"""
Tests for performance presets (Fase 1).

Covers:
  - 3 presets are defined (eco, balanced, boost)
  - set_mode applies all preset knobs atomically
  - set_mode rejects unknown names
  - Auto-mode promotes/demotes based on CPU usage
  - Auto-mode respects hysteresis
  - Auto-mode can be enabled/disabled at runtime
"""

import os
import sys
import unittest
import threading
import time
import psutil  # for tick() in AutoMode tests
from unittest.mock import MagicMock, patch
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "train"))


def _make_trainer(tmpdir: str):
    """Reuse the helper from test_hot_swap."""
    from train.league.league_trainer import LeagueTrainer
    from train.league.replay_buffer import ReplayBuffer

    trainer = LeagueTrainer.__new__(LeagueTrainer)
    trainer.device = MagicMock()
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

    # Set up performance mode infrastructure (mirrors __init__ tail)
    from train.league.performance import AutoModeConfig, AutoModeController
    trainer.performance_mode = "balanced"
    trainer._auto_mode = AutoModeController(trainer, AutoModeConfig(enabled=False))
    return trainer


class PresetDefinitionTests(unittest.TestCase):
    """The preset table itself."""

    def test_four_presets_exist(self):
        from train.league.performance import PRESETS, list_preset_names
        names = list_preset_names()
        self.assertEqual(set(names), {"low_memory", "eco", "balanced", "boost"})

    def test_balanced_matches_class_defaults(self):
        from train.league.performance import PRESETS
        bal = PRESETS["balanced"]
        # Match the class-level defaults defined in LeagueTrainer
        self.assertEqual(bal.batch_size, 256)
        self.assertEqual(bal.mcts_visits_selfplay, 200)
        self.assertEqual(bal.num_self_play_workers, 6)
        self.assertEqual(bal.games_per_worker_per_round, 5)

    def test_preset_ordering_for_safety(self):
        """low_memory <= eco <= balanced <= boost (in resource usage)."""
        from train.league.performance import PRESETS
        low_mem = PRESETS["low_memory"]
        eco = PRESETS["eco"]
        bal = PRESETS["balanced"]
        boost = PRESETS["boost"]
        self.assertLess(low_mem.batch_size, eco.batch_size)
        self.assertLess(eco.batch_size, bal.batch_size)
        self.assertLess(bal.batch_size, boost.batch_size)
        self.assertLess(low_mem.mcts_visits_selfplay, eco.mcts_visits_selfplay)
        self.assertLess(eco.mcts_visits_selfplay, bal.mcts_visits_selfplay)
        self.assertLess(bal.mcts_visits_selfplay, boost.mcts_visits_selfplay)
        self.assertLessEqual(low_mem.num_self_play_workers, eco.num_self_play_workers)
        self.assertLessEqual(eco.num_self_play_workers, bal.num_self_play_workers)
        self.assertLessEqual(bal.num_self_play_workers, boost.num_self_play_workers)
        # Memory budget: buffer size strictly increases
        self.assertLess(low_mem.replay_buffer_max_size, eco.replay_buffer_max_size)
        self.assertLess(eco.replay_buffer_max_size, bal.replay_buffer_max_size)
        self.assertLess(bal.replay_buffer_max_size, boost.replay_buffer_max_size)

    def test_preset_as_knob_dict_has_all_preset_knobs(self):
        from train.league.performance import PRESETS, PRESET_KNOBS
        for name, preset in PRESETS.items():
            d = preset.as_knob_dict()
            self.assertEqual(set(d.keys()), set(PRESET_KNOBS),
                             f"Preset '{name}' knobs mismatch")

    def test_get_preset_unknown_raises(self):
        from train.league.performance import get_preset
        with self.assertRaises(KeyError):
            get_preset("nuclear")


class SetModeTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)

    def test_set_mode_balanced_applies_defaults(self):
        self.assertTrue(self.trainer.set_mode("balanced"))
        self.assertEqual(self.trainer.performance_mode, "balanced")
        # BATCH_SIZE is immediate
        self.assertEqual(self.trainer.BATCH_SIZE, 256)
        # MCTS visits is deferred
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer.MCTS_VISITS_SELFPLAY, 200)

    def test_set_mode_eco_reduces_batch(self):
        self.assertTrue(self.trainer.set_mode("eco"))
        # BATCH_SIZE is immediate
        self.assertEqual(self.trainer.BATCH_SIZE, 128)
        self.assertEqual(self.trainer.GAMES_PER_WORKER_PER_ROUND, 2)
        # NUM_SELF_PLAY_WORKERS is deferred
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer.NUM_SELF_PLAY_WORKERS, 3)
        self.assertEqual(self.trainer._num_self_play_workers, 3)

    def test_set_mode_boost_increases_batch(self):
        self.assertTrue(self.trainer.set_mode("boost"))
        self.assertEqual(self.trainer.BATCH_SIZE, 1024)
        self.assertEqual(self.trainer.GAMES_PER_WORKER_PER_ROUND, 8)
        # Verify the deferred values are queued
        self.assertIn("NUM_SELF_PLAY_WORKERS", self.trainer._pending_changes)
        self.trainer._apply_pending_changes()
        self.assertEqual(self.trainer.NUM_SELF_PLAY_WORKERS, 12)
        self.assertEqual(self.trainer._num_self_play_workers, 12)

    def test_set_mode_unknown_returns_false(self):
        self.assertFalse(self.trainer.set_mode("nuclear"))
        # Performance mode unchanged
        self.assertEqual(self.trainer.performance_mode, "balanced")

    def test_get_mode(self):
        self.assertEqual(self.trainer.get_mode(), "balanced")
        self.trainer.set_mode("eco")
        self.assertEqual(self.trainer.get_mode(), "eco")

    def test_list_available_modes(self):
        modes = self.trainer.list_available_modes()
        self.assertIn("eco", modes)
        self.assertIn("balanced", modes)
        self.assertIn("boost", modes)

    def test_describe_mode_returns_knobs(self):
        d = self.trainer.describe_mode("boost")
        self.assertEqual(d["BATCH_SIZE"], 1024)
        self.assertEqual(d["MCTS_VISITS_SELFPLAY"], 400)

    def test_describe_mode_unknown_empty(self):
        self.assertEqual(self.trainer.describe_mode("nuclear"), {})

    def test_set_mode_preserves_non_preset_knobs(self):
        """Knobs not in PRESET_KNOBS should NOT be touched by set_mode."""
        # Pick something not in the preset list, e.g. POLICY_LOSS_WEIGHT
        self.trainer.POLICY_LOSS_WEIGHT = 7.0
        self.trainer.set_mode("boost")
        self.assertEqual(self.trainer.POLICY_LOSS_WEIGHT, 7.0)


class AutoModeTests(unittest.TestCase):

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.trainer = _make_trainer(self.tmp)
        # Speed up the test by overriding intervals
        from train.league.performance import AutoModeConfig
        self.trainer._auto_mode.config.poll_interval_sec = 0.05
        self.trainer._auto_mode.config.sample_interval_sec = 0.0
        self.trainer._auto_mode.config.min_seconds_between_changes = 0.0

    def tearDown(self):
        self.trainer.set_auto_mode(False)

    def test_auto_mode_disabled_by_default(self):
        self.assertFalse(self.trainer.get_auto_mode())

    def test_set_auto_mode_starts_watchdog(self):
        self.trainer.set_auto_mode(True)
        self.assertTrue(self.trainer.get_auto_mode())
        # Thread should be alive
        self.assertIsNotNone(self.trainer._auto_mode._thread)
        self.assertTrue(self.trainer._auto_mode._thread.is_alive())

    def test_auto_mode_can_be_disabled(self):
        self.trainer.set_auto_mode(True)
        self.trainer.set_auto_mode(False)
        self.assertFalse(self.trainer.get_auto_mode())
        # After stop(), _thread is set to None
        time.sleep(0.2)
        self.assertIsNone(self.trainer._auto_mode._thread)

    def test_auto_mode_promotes_when_cpu_idle(self):
        """CPU=5% (low) with current=eco => should promote to balanced."""
        self.trainer.set_mode("eco")
        self.trainer._auto_mode.config.promote_cpu_pct = 20.0
        self.trainer._auto_mode.config.demote_cpu_pct = 60.0
        # Trigger a single tick directly (avoiding the polling loop's wait)
        with patch("psutil.cpu_percent", return_value=5.0):
            self.trainer._auto_mode._tick(psutil)
        self.assertEqual(self.trainer.get_mode(), "balanced")

    def test_auto_mode_demotes_when_cpu_busy(self):
        """CPU=80% (high) with current=boost => should demote to balanced."""
        self.trainer.set_mode("boost")
        self.trainer._auto_mode.config.promote_cpu_pct = 20.0
        self.trainer._auto_mode.config.demote_cpu_pct = 60.0
        with patch("psutil.cpu_percent", return_value=80.0):
            self.trainer._auto_mode._tick(psutil)
        self.assertEqual(self.trainer.get_mode(), "balanced")

    def test_auto_mode_demotes_further_when_cpu_high(self):
        """CPU=90% with current=boost => boost -> balanced -> eco."""
        self.trainer.set_mode("boost")
        self.trainer._auto_mode.config.promote_cpu_pct = 20.0
        self.trainer._auto_mode.config.demote_cpu_pct = 60.0
        self.trainer._auto_mode.config.min_seconds_between_changes = 0.0
        with patch("psutil.cpu_percent", return_value=90.0):
            self.trainer._auto_mode._tick(psutil)
            # 2nd tick would normally require hysteresis; we already set it to 0
            self.trainer._auto_mode._tick(psutil)
        # Should have stepped all the way down
        self.assertEqual(self.trainer.get_mode(), "eco")

    def test_auto_mode_no_change_in_safe_zone(self):
        """CPU=30% (between promote and demote thresholds) => no change."""
        self.trainer.set_mode("balanced")
        self.trainer._auto_mode.config.promote_cpu_pct = 20.0
        self.trainer._auto_mode.config.demote_cpu_pct = 60.0
        with patch("psutil.cpu_percent", return_value=30.0):
            self.trainer._auto_mode._tick(psutil)
        self.assertEqual(self.trainer.get_mode(), "balanced")

    def test_auto_mode_hysteresis(self):
        """Within min_seconds_between_changes, no change even if CPU spikes."""
        self.trainer.set_mode("boost")
        self.trainer._auto_mode.config.promote_cpu_pct = 20.0
        self.trainer._auto_mode.config.demote_cpu_pct = 60.0
        # Re-enable hysteresis (setUp set it to 0)
        self.trainer._auto_mode.config.min_seconds_between_changes = 300.0
        # Force a recent change timestamp so we're inside the hysteresis window
        self.trainer._auto_mode._last_change_time = time.time()
        with patch("psutil.cpu_percent", return_value=90.0):
            self.trainer._auto_mode._tick(psutil)
        # Should NOT change because hysteresis window hasn't elapsed
        self.assertEqual(self.trainer.get_mode(), "boost")

    def test_auto_mode_at_top_boundary_does_not_promote(self):
        """At top of order (boost), promote should be a no-op."""
        self.trainer.set_mode("boost")
        self.trainer._auto_mode.config.promote_cpu_pct = 50.0
        self.trainer._auto_mode.config.demote_cpu_pct = 80.0
        with patch("psutil.cpu_percent", return_value=5.0):
            self.trainer._auto_mode._tick(psutil)
        # No order to go higher (boost is the highest)
        self.assertEqual(self.trainer.get_mode(), "boost")

    def test_auto_mode_at_bottom_boundary_does_not_demote(self):
        """At bottom of order (low_memory), demote should be a no-op."""
        self.trainer.set_mode("low_memory")
        self.trainer._auto_mode.config.promote_cpu_pct = 10.0
        self.trainer._auto_mode.config.demote_cpu_pct = 50.0
        with patch("psutil.cpu_percent", return_value=99.0):
            self.trainer._auto_mode._tick(psutil)
        # No order to go lower (low_memory is the lowest)
        self.assertEqual(self.trainer.get_mode(), "low_memory")


if __name__ == "__main__":
    unittest.main()
