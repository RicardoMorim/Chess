"""
Tests for the worker-count hot-swap and the new ``gpu_self_play_workers``
preset field.

Verifies:
  * ``GPU_SELF_PLAY_WORKERS`` default is reduced from 14 to 8
  * ``GPU_SELF_PLAY_WORKERS`` is in ``_HOT_KNOBS_DEFERRED`` and hot-settable
  * The set_knob pipeline routes both NUM_SELF_PLAY_WORKERS and
    GPU_SELF_PLAY_WORKERS correctly when GPU batching is on
  * All four presets declare a ``gpu_self_play_workers`` value
  * Process count and worker RAM estimates are sensible
  * RAM throttling uses the correct worker count per mode
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.league.performance import PRESETS, PerformancePreset


class GPUWorkerDefaultTests(unittest.TestCase):
    """The default GPU worker count must be lowered for low-RAM laptops."""

    def test_default_gpu_workers_is_at_most_8(self):
        """Was 14, now 8. Keeps total processes (8 x 3 = 24) under control."""
        # Import here to avoid pulling LeagueTrainer deps in the preset-only tests
        from train.league.league_trainer import LeagueTrainer
        self.assertLessEqual(LeagueTrainer.GPU_SELF_PLAY_WORKERS, 8)

    def test_default_gpu_workers_fits_32gb(self):
        """8 workers x 3 variants = 24 processes x ~400 MB = ~9.6 GB in workers."""
        from train.league.league_trainer import LeagueTrainer
        per_process_mb = 400
        processes = LeagueTrainer.GPU_SELF_PLAY_WORKERS * 3
        ram_gb = processes * per_process_mb / 1024
        self.assertLess(ram_gb, 12.0, f"workers use {ram_gb:.1f} GB, expected <12")


class GPUWorkerHotSwapTests(unittest.TestCase):
    """GPU_SELF_PLAY_WORKERS is hot-settable via set_knob()."""

    def test_gpu_workers_in_hot_knobs(self):
        from train.league.league_trainer import LeagueTrainer
        self.assertIn("GPU_SELF_PLAY_WORKERS", LeagueTrainer._HOT_KNOBS)

    def test_gpu_workers_in_deferred_bucket(self):
        from train.league.league_trainer import LeagueTrainer
        self.assertIn("GPU_SELF_PLAY_WORKERS", LeagueTrainer._HOT_KNOBS_DEFERRED)

    def test_set_knob_gpu_workers_queues_change(self):
        from train.league.league_trainer import LeagueTrainer
        trainer = LeagueTrainer.__new__(LeagueTrainer)
        trainer._state_lock = __import__("threading").RLock()
        trainer._pending_changes = {}
        trainer.performance_mode = "balanced"
        trainer.use_gpu_batching = True
        trainer._num_self_play_workers = 8
        # Patch __init__-dependent attrs
        trainer.NUM_SELF_PLAY_WORKERS = 6
        trainer.GPU_SELF_PLAY_WORKERS = 8
        trainer.REPLAY_BUFFER_MAX_SIZE = 100_000
        trainer._buffer_target_size = 100_000
        trainer._last_buffer_target_size = 100_000
        trainer.MCTS_VISITS_SELFPLAY = 200
        trainer._current_mcts_visits = 200
        trainer.MCTS_VISITS_EVAL = 400
        trainer.evaluator = None
        trainer.SELF_PLAY_VARIANT_PARALLELISM = 3
        trainer._variant_parallelism = 3
        trainer.buffers = {}
        trainer.VARIANTS = ("baseline", "attack", "est")
        # The change is queued (deferred)
        ok = trainer.set_knob("GPU_SELF_PLAY_WORKERS", 4)
        self.assertTrue(ok)
        self.assertIn("GPU_SELF_PLAY_WORKERS", trainer._pending_changes)
        self.assertEqual(trainer._pending_changes["GPU_SELF_PLAY_WORKERS"], 4)
        # Apply pending changes
        trainer._apply_pending_changes()
        self.assertEqual(trainer._num_self_play_workers, 4)
        self.assertEqual(trainer.GPU_SELF_PLAY_WORKERS, 4)

    def test_set_knob_num_workers_with_gpu_batching_uses_max(self):
        """When GPU batching is on, NUM_SELF_PLAY_WORKERS raise should also
        raise GPU_SELF_PLAY_WORKERS (or vice versa), so the GPU pipeline
        stays fed even if the user only set one of them."""
        from train.league.league_trainer import LeagueTrainer
        import threading as _t
        trainer = LeagueTrainer.__new__(LeagueTrainer)
        trainer._state_lock = _t.RLock()
        trainer._pending_changes = {}
        trainer.performance_mode = "balanced"
        trainer.use_gpu_batching = True
        trainer._num_self_play_workers = 8
        trainer.NUM_SELF_PLAY_WORKERS = 6
        trainer.GPU_SELF_PLAY_WORKERS = 8
        # Apply NUM_SELF_PLAY_WORKERS=10 — should result in 10 workers
        # (max(8, 10) = 10)
        trainer.set_knob("NUM_SELF_PLAY_WORKERS", 10)
        trainer._apply_pending_changes()
        self.assertEqual(trainer._num_self_play_workers, 10)
        # And GPU_SELF_PLAY_WORKERS=4 with NUM_SELF_PLAY_WORKERS=6 (default)
        trainer.set_knob("GPU_SELF_PLAY_WORKERS", 4)
        trainer._apply_pending_changes()
        self.assertEqual(trainer._num_self_play_workers, 4)


class PresetGPUWorkerTests(unittest.TestCase):
    """Every preset must declare gpu_self_play_workers."""

    def test_all_presets_declare_field(self):
        for name, preset in PRESETS.items():
            self.assertIsInstance(
                preset.gpu_self_play_workers, int,
                f"Preset {name} missing gpu_self_play_workers",
            )
            self.assertGreaterEqual(preset.gpu_self_play_workers, 1)

    def test_preset_knobs_include_gpu_workers(self):
        from train.league.performance import PRESET_KNOBS
        self.assertIn("GPU_SELF_PLAY_WORKERS", PRESET_KNOBS)

    def test_as_knob_dict_includes_gpu_workers(self):
        for name, preset in PRESETS.items():
            d = preset.as_knob_dict()
            self.assertIn("GPU_SELF_PLAY_WORKERS", d)
            self.assertEqual(d["GPU_SELF_PLAY_WORKERS"], preset.gpu_self_play_workers)

    def test_low_memory_has_fewest_gpu_workers(self):
        low = PRESETS["low_memory"]
        for name, preset in PRESETS.items():
            if name == "low_memory":
                continue
            self.assertLessEqual(
                low.gpu_self_play_workers,
                preset.gpu_self_play_workers,
            )

    def test_worker_count_monotonic_with_resource_use(self):
        """low_memory < eco < balanced < boost in gpu workers."""
        self.assertLess(PRESETS["low_memory"].gpu_self_play_workers, PRESETS["eco"].gpu_self_play_workers)
        self.assertLessEqual(PRESETS["eco"].gpu_self_play_workers, PRESETS["balanced"].gpu_self_play_workers)
        self.assertLess(PRESETS["balanced"].gpu_self_play_workers, PRESETS["boost"].gpu_self_play_workers)


class PresetProcessCountTests(unittest.TestCase):
    """Process count and worker RAM estimates are sensible."""

    def test_balanced_total_processes_under_30(self):
        """8 GPU workers x 3 variants = 24 worker processes + 1 main = 25."""
        bal = PRESETS["balanced"]
        # estimated_process_count uses gpu_self_play_workers
        self.assertEqual(bal.estimated_process_count(), 8 * 3 + 1)

    def test_low_memory_total_processes_under_15(self):
        low = PRESETS["low_memory"]
        # 4 GPU workers x 2 variants + 1 main = 9
        self.assertLessEqual(low.estimated_process_count(), 4 * 2 + 1)

    def test_boost_total_processes_above_40(self):
        boost = PRESETS["boost"]
        # 14 GPU workers x 3 variants + 1 main = 43
        self.assertGreaterEqual(boost.estimated_process_count(), 14 * 3 + 1)

    def test_worker_ram_estimate_uses_per_process_mb(self):
        bal = PRESETS["balanced"]
        # 8 GPU workers x 3 variants + 1 main = 25 processes
        # 25 * 400 MB / 1024 = 9.77 GB
        self.assertAlmostEqual(bal.estimated_worker_ram_gb(per_process_mb=400), 9.77, places=1)

    def test_low_memory_worker_ram_under_5gb(self):
        low = PRESETS["low_memory"]
        # 4 GPU workers x 2 variants x 400 MB / 1024 = 3.125 GB
        self.assertLess(low.estimated_worker_ram_gb(per_process_mb=400), 5.0)


class SetModeRespectsGPUWorkersTests(unittest.TestCase):
    """End-to-end: set_mode() routes GPU_SELF_PLAY_WORKERS through the pipeline."""

    def _make_minimal_trainer(self):
        """Build a trainer with enough attrs to exercise set_mode/set_knob."""
        from train.league.league_trainer import LeagueTrainer
        import threading as _t
        t = LeagueTrainer.__new__(LeagueTrainer)
        t._state_lock = _t.RLock()
        t._pending_changes = {}
        t.performance_mode = "balanced"
        t.use_gpu_batching = True
        t._num_self_play_workers = 8
        t._variant_parallelism = 3
        t._buffer_target_size = 100_000
        t._last_buffer_target_size = 100_000
        t._current_mcts_visits = 200
        t.evaluator = None
        t.buffers = {}
        t.VARIANTS = ("baseline", "attack", "est")
        # Class-level knob defaults (what presets override)
        t.BATCH_SIZE = 256
        t.TRAINING_STEPS_PER_ROUND = 50
        t.GAMES_PER_WORKER_PER_ROUND = 5
        t.MCTS_VISITS_SELFPLAY = 200
        t.MCTS_VISITS_EVAL = 400
        t.NUM_SELF_PLAY_WORKERS = 6
        t.GPU_SELF_PLAY_WORKERS = 8
        t.REPLAY_BUFFER_MAX_SIZE = 100_000
        t.GPU_INFER_BATCH_SIZE = 64
        t.STOCKFISH_BENCH_EVERY_N_ROUNDS = 25
        t.PUZZLE_BATCHES_PER_GAME_BATCH = 1
        t.PROGAME_BATCHES_PER_GAME_BATCH = 0
        t.SELF_PLAY_VARIANT_PARALLELISM = 3
        return t

    def test_set_mode_low_memory_queues_gpu_worker_reduction(self):
        t = self._make_minimal_trainer()
        ok = t.set_mode("low_memory")
        self.assertTrue(ok)
        self.assertEqual(t.performance_mode, "low_memory")
        # Apply pending changes
        t._apply_pending_changes()
        self.assertEqual(t._num_self_play_workers, 4)  # low_memory GPU workers

    def test_set_mode_balanced_uses_8_gpu_workers(self):
        t = self._make_minimal_trainer()
        t.set_mode("balanced")
        t._apply_pending_changes()
        self.assertEqual(t._num_self_play_workers, 8)

    def test_set_mode_boost_uses_14_gpu_workers(self):
        t = self._make_minimal_trainer()
        t.set_mode("boost")
        t._apply_pending_changes()
        self.assertEqual(t._num_self_play_workers, 14)


if __name__ == "__main__":
    unittest.main()
