"""
Tests for the new compact-array ReplayBuffer and the ``low_memory`` preset.

Verifies:
  * Memory footprint is dramatically smaller than the old object-dtype version
  * Channel/policy padding works for 18-channel tensors into a 22-channel buffer
  * Live ``set_max_size`` shrink preserves the most recent entries
  * Legacy add_game / sample list-of-arrays API still works
  * New fast-path ``sample(return_numpy=True)`` returns batched arrays
  * ``add_many`` bulk copy is correct
  * ``low_memory`` preset values are coherent and below ``eco``
  * The preset order is monotonic in resource usage
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

# Make ``train`` importable when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.league.replay_buffer import ReplayBuffer
from train.league.performance import PRESETS


def _rand_pos(channels: int = 22) -> np.ndarray:
    """Random position tensor of shape (channels, 8, 8) fp32 in [-1, 1]."""
    return np.random.randn(channels, 8, 8).astype(np.float32)


def _rand_policy(size: int = 4096) -> np.ndarray:
    return np.random.rand(size).astype(np.float32)


class ReplayBufferMemoryFootprintTests(unittest.TestCase):
    """The compact buffer should be much smaller than the legacy object dtype."""

    def test_compact_buffer_uses_fixed_flat_arrays(self):
        buf = ReplayBuffer(max_size=1000, pos_channels=22, policy_size=4096)
        # Sanity: pre-allocated storage exists with the expected dtype.
        self.assertEqual(buf._positions.shape, (1000, 22, 8, 8))
        self.assertEqual(buf._policies.shape, (1000, 4096))
        self.assertEqual(buf._values.shape, (1000,))
        self.assertEqual(buf._positions.dtype, np.float16)
        self.assertEqual(buf._policies.dtype, np.float16)
        self.assertEqual(buf._values.dtype, np.float32)
        # No Python objects held per entry (the legacy design had
        # ``np.empty(max_size, dtype=object)`` which is ~50-100x larger).
        total_bytes = (
            buf._positions.nbytes + buf._policies.nbytes + buf._values.nbytes
        )
        # 100K x (22x8x8x2 + 4096x2 + 4) bytes = ~3.1 GB
        # Per-entry budget: ~10.5 KB
        per_entry_kb = total_bytes / 1000 / 1024
        self.assertLess(per_entry_kb, 11.0, f"per-entry {per_entry_kb:.2f}KB too large")
        self.assertGreater(per_entry_kb, 9.0, f"per-entry {per_entry_kb:.2f}KB suspiciously small")

    def test_no_object_dtype_storage(self):
        """Regression: must not regress to the legacy np.empty(dtype=object)."""
        buf = ReplayBuffer(max_size=100)
        self.assertNotEqual(buf._positions.dtype, object)
        self.assertNotEqual(buf._policies.dtype, object)


class ReplayBufferChannelPaddingTests(unittest.TestCase):
    """18-channel baseline data must fit into a 22-channel buffer without crashing."""

    def test_smaller_channel_position_is_padded(self):
        buf = ReplayBuffer(max_size=10, pos_channels=22, policy_size=4096)
        small_pos = _rand_pos(channels=18)
        buf.add_game([(small_pos, _rand_policy(), 0.5)])
        self.assertEqual(len(buf), 1)
        # Stored tensor is full 22 channels
        self.assertEqual(buf._positions[0].shape, (22, 8, 8))
        # First 18 channels match the input
        np.testing.assert_array_equal(
            buf._positions[0][:18].astype(np.float32),
            small_pos.astype(np.float16).astype(np.float32),
        )
        # Last 4 channels are zero-padded
        np.testing.assert_array_equal(
            buf._positions[0][18:].astype(np.float32),
            np.zeros((4, 8, 8), dtype=np.float32),
        )

    def test_larger_channel_position_is_truncated(self):
        buf = ReplayBuffer(max_size=10, pos_channels=18, policy_size=4096)
        big_pos = _rand_pos(channels=22)
        buf.add_game([(big_pos, _rand_policy(), 0.5)])
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf._positions[0].shape, (18, 8, 8))

    def test_smaller_policy_is_padded(self):
        buf = ReplayBuffer(max_size=10, pos_channels=22, policy_size=4096)
        small_policy = _rand_policy(size=1024)
        buf.add_game([(_rand_pos(22), small_policy, 0.0)])
        # Stored policy is full size; first 1024 entries match the input
        np.testing.assert_array_equal(
            buf._policies[0][:1024].astype(np.float32),
            small_policy.astype(np.float16).astype(np.float32),
        )
        np.testing.assert_array_equal(
            buf._policies[0][1024:].astype(np.float32),
            np.zeros(4096 - 1024, dtype=np.float32),
        )

    def test_larger_policy_is_truncated(self):
        buf = ReplayBuffer(max_size=10, pos_channels=22, policy_size=4096)
        big_policy = _rand_policy(size=8192)
        buf.add_game([(_rand_pos(22), big_policy, 0.0)])
        self.assertEqual(buf._policies[0].shape, (4096,))


class ReplayBufferSampleTests(unittest.TestCase):
    """Both legacy list-of-arrays and fast-path batched array APIs work."""

    def setUp(self):
        self.buf = ReplayBuffer(max_size=100, pos_channels=22, policy_size=4096)
        game = [(_rand_pos(22), _rand_policy(), float(i) * 0.01) for i in range(20)]
        self.buf.add_game(game)

    def test_legacy_list_of_arrays(self):
        positions, policies, values = self.buf.sample(5, return_numpy=False)
        self.assertEqual(len(positions), 5)
        self.assertEqual(len(policies), 5)
        self.assertEqual(len(values), 5)
        for p in positions:
            self.assertEqual(p.shape, (22, 8, 8))
            self.assertEqual(p.dtype, np.float32)
        for p in policies:
            self.assertEqual(p.shape, (4096,))
            self.assertEqual(p.dtype, np.float32)
        for v in values:
            self.assertIsInstance(v, float)

    def test_fast_path_batched_arrays(self):
        positions, policies, values = self.buf.sample(5, return_numpy=True)
        self.assertEqual(positions.shape, (5, 22, 8, 8))
        self.assertEqual(policies.shape, (5, 4096))
        self.assertEqual(values.shape, (5,))
        self.assertEqual(positions.dtype, np.float32)
        self.assertEqual(values.dtype, np.float32)

    def test_sample_larger_than_buffer_errors(self):
        with self.assertRaises(ValueError):
            self.buf.sample(1000)

    def test_stats_running_sum_is_accurate(self):
        stats = self.buf.get_stats()
        self.assertEqual(stats["size"], 20)
        self.assertEqual(stats["capacity"], 100)
        self.assertAlmostEqual(stats["fill_ratio"], 0.2, places=4)
        # Sanity: mean of [0..0.19] in 0.01 steps = 0.095
        self.assertAlmostEqual(stats["value_mean"], 0.095, places=3)


class ReplayBufferAddManyTests(unittest.TestCase):
    """Bulk add_many is correct and matches add_game semantics."""

    def test_add_many_matches_add_game(self):
        buf_a = ReplayBuffer(max_size=100)
        buf_b = ReplayBuffer(max_size=100)
        n = 25
        positions = [_rand_pos(22) for _ in range(n)]
        policies = [_rand_policy() for _ in range(n)]
        values = np.linspace(-1, 1, n).astype(np.float32)
        buf_a.add_many(positions, policies, values)
        for pos, pol, v in zip(positions, policies, values):
            buf_b.add_game([(pos, pol, float(v))])
        self.assertEqual(len(buf_a), len(buf_b))
        a_pos, a_pol, a_val = buf_a.sample(n, return_numpy=True)
        b_pos, b_pol, b_val = buf_b.sample(n, return_numpy=True)
        # Same content, possibly different order due to random.sample
        a_keys = sorted(a_val.tolist())
        b_keys = sorted(b_val.tolist())
        np.testing.assert_array_almost_equal(a_keys, b_keys, decimal=3)

    def test_add_many_length_mismatch_errors(self):
        buf = ReplayBuffer(max_size=10)
        with self.assertRaises(ValueError):
            buf.add_many([_rand_pos(22)], [_rand_policy(), _rand_policy()], [0.0])

    def test_add_many_empty(self):
        buf = ReplayBuffer(max_size=10)
        buf.add_many([], [], [])
        self.assertEqual(len(buf), 0)


class ReplayBufferSetMaxSizeTests(unittest.TestCase):
    """Live shrink/grow preserves data in chronological order."""

    def setUp(self):
        self.buf = ReplayBuffer(max_size=100, pos_channels=22, policy_size=4096)
        for i in range(50):
            self.buf.add_game([(_rand_pos(22), _rand_policy(), float(i) * 0.1)])

    def test_shrink_keeps_most_recent(self):
        self.buf.set_max_size(20)
        self.assertEqual(self.buf.max_size, 20)
        self.assertEqual(len(self.buf), 20)
        # Most recent values are 30..49 (we added 0..49, kept last 20)
        # Stats mean should match what we kept
        stats = self.buf.get_stats()
        expected_mean = np.mean([i * 0.1 for i in range(30, 50)])
        self.assertAlmostEqual(stats["value_mean"], expected_mean, places=3)

    def test_grow_preserves_data(self):
        self.buf.set_max_size(200)
        self.assertEqual(self.buf.max_size, 200)
        self.assertEqual(len(self.buf), 50)

    def test_shrink_to_empty(self):
        self.buf.set_max_size(0)
        # new_max_size <= 0 is a no-op
        self.assertEqual(self.buf.max_size, 100)

    def test_shrink_to_smaller_than_size(self):
        # Buffer has 50 entries, shrink to 10 (keeps last 10)
        self.buf.set_max_size(10)
        self.assertEqual(len(self.buf), 10)

    def test_shrink_to_zero_data(self):
        empty = ReplayBuffer(max_size=100)
        empty.set_max_size(20)
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.max_size, 20)

    def test_set_max_size_to_same_value_noop(self):
        self.buf.set_max_size(100)
        self.assertEqual(len(self.buf), 50)

    def test_overwrite_after_shrink(self):
        self.buf.set_max_size(10)
        # Add 15 more entries; should overwrite the oldest
        for i in range(15):
            self.buf.add_game([(_rand_pos(22), _rand_policy(), -0.5)])
        self.assertEqual(len(self.buf), 10)
        stats = self.buf.get_stats()
        # All values are -0.5 now (overwrites)
        self.assertAlmostEqual(stats["value_mean"], -0.5, places=3)


class ReplayBufferPersistenceTests(unittest.TestCase):
    """save_to_npz / load_from_npz round-trips correctly."""

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = ReplayBuffer(max_size=100)
            for i in range(30):
                buf.add_game([(_rand_pos(22), _rand_policy(), float(i))])
            path = os.path.join(tmp, "buf.npz")
            buf.save_to_npz(path)
            self.assertTrue(os.path.exists(path))

            buf2 = ReplayBuffer(max_size=100)
            buf2.load_from_npz(path)
            self.assertEqual(len(buf2), 30)
            np.testing.assert_array_equal(
                sorted(buf2._values[:30]),
                sorted(buf._values[:30]),
            )

    def test_load_smaller_buffer_truncates(self):
        with tempfile.TemporaryDirectory() as tmp:
            buf = ReplayBuffer(max_size=100)
            for i in range(50):
                buf.add_game([(_rand_pos(22), _rand_policy(), float(i))])
            path = os.path.join(tmp, "buf.npz")
            buf.save_to_npz(path)
            buf2 = ReplayBuffer(max_size=20)
            buf2.load_from_npz(path)
            # Truncated to most recent 20
            self.assertEqual(len(buf2), 20)
            np.testing.assert_array_equal(
                sorted(buf2._values[:20]),
                sorted(buf._values[30:50]),
            )


class LowMemoryPresetTests(unittest.TestCase):
    """The new ``low_memory`` preset must exist and be strictly below ``eco``."""

    def test_low_memory_preset_exists(self):
        self.assertIn("low_memory", PRESETS)

    def test_low_memory_has_lowest_buffer_size(self):
        low = PRESETS["low_memory"]
        for name, preset in PRESETS.items():
            if name == "low_memory":
                continue
            self.assertLess(
                low.replay_buffer_max_size,
                preset.replay_buffer_max_size,
                f"low_memory buffer should be smaller than {name}",
            )

    def test_low_memory_has_lowest_workers(self):
        low = PRESETS["low_memory"]
        self.assertLessEqual(low.num_self_play_workers, PRESETS["eco"].num_self_play_workers)

    def test_low_memory_has_fewer_visits_than_eco(self):
        low = PRESETS["low_memory"]
        self.assertLess(low.mcts_visits_selfplay, PRESETS["eco"].mcts_visits_selfplay)

    def test_low_memory_disables_puzzle_batches(self):
        low = PRESETS["low_memory"]
        self.assertEqual(low.puzzle_batches_per_game_batch, 0)

    def test_low_memory_estimated_memory_below_1gb(self):
        """Three variants x 20K x ~10.5KB ~= 0.6 GB. Must stay under 1 GB."""
        low = PRESETS["low_memory"]
        per_entry_bytes = 22 * 8 * 8 * 2 + 4096 * 2 + 4  # fp16 pos+pol, fp32 val
        total_gb = (low.replay_buffer_max_size * 3 * per_entry_bytes) / (1024 ** 3)
        self.assertLess(total_gb, 1.0, f"low_memory uses {total_gb:.2f} GB, expected <1")

    def test_eco_estimated_memory_below_2gb(self):
        eco = PRESETS["eco"]
        per_entry_bytes = 22 * 8 * 8 * 2 + 4096 * 2 + 4
        total_gb = (eco.replay_buffer_max_size * 3 * per_entry_bytes) / (1024 ** 3)
        self.assertLess(total_gb, 2.0, f"eco uses {total_gb:.2f} GB, expected <2")

    def test_preset_buffer_size_ordering(self):
        sizes = {n: p.replay_buffer_max_size for n, p in PRESETS.items()}
        self.assertEqual(
            sizes["low_memory"],
            min(sizes.values()),
        )
        self.assertEqual(
            sizes["boost"],
            max(sizes.values()),
        )


if __name__ == "__main__":
    unittest.main()
