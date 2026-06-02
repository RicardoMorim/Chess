"""
Persistent Replay Buffer for League Training
=============================================

This module implements a FIFO replay buffer that stores game data
from self-play for training. Each model variant has its own buffer
to prevent catastrophic forgetting and maintain learning stability.

Design principles:
- Per-model FIFO buffers (separate data streams)
- Capped size to prevent memory explosion
- NumPy circular buffer: all operations O(1) or O(batch_size)
- No mixing of data between buffer and opponents (decoupling)
- Thread-safe operations for multi-process workers
"""

import random
import threading
from typing import List, Tuple, Optional
import numpy as np


class ReplayBuffer:
    """
    A FIFO replay buffer backed by a NumPy circular buffer.

    Thread-safe for concurrent adds and samples.
    Automatically overwrites oldest entries when max capacity is reached.

    Each entry: (position_tensor, policy_vector, value_scalar)

    All operations are O(1) except sample() which is O(batch_size) and
    get_stats() which is O(size).
    """

    def __init__(self, max_size: int = 200_000):
        self.max_size = max_size
        self.lock = threading.Lock()
        self._positions = np.empty(max_size, dtype=object)
        self._policies = np.empty(max_size, dtype=object)
        self._values = np.zeros(max_size, dtype=np.float32)
        self._size = 0
        self._head = 0

    def _logical_to_physical(self, logical_idx: int) -> int:
        """Convert a logical index (0 = oldest) to physical array index."""
        if self._size < self.max_size:
            return logical_idx
        return (self._head + logical_idx) % self.max_size

    def add_game(self, game_trajectory: List[Tuple]) -> None:
        with self.lock:
            for position, policy, value in game_trajectory:
                self._positions[self._head] = np.array(position, dtype=np.float16)
                self._policies[self._head] = np.array(policy, dtype=np.float16)
                self._values[self._head] = np.float32(value)
                self._head = (self._head + 1) % self.max_size
                if self._size < self.max_size:
                    self._size += 1

    def sample(self, batch_size: int) -> Tuple[List, List, List]:
        with self.lock:
            if batch_size > self._size:
                raise ValueError(
                    f"Batch size {batch_size} exceeds buffer size {self._size}"
                )

            logical_indices = random.sample(range(self._size), batch_size)
            physical = [self._logical_to_physical(i) for i in logical_indices]

            positions = [np.array(self._positions[i], dtype=np.float32) for i in physical]
            policies = [np.array(self._policies[i], dtype=np.float32) for i in physical]
            values = [self._values[i] for i in physical]

        return positions, policies, values

    def __len__(self) -> int:
        with self.lock:
            return self._size

    def is_ready(self, min_size: int = 256) -> bool:
        return len(self) >= min_size

    def clear(self) -> None:
        with self.lock:
            self._size = 0
            self._head = 0

    def set_max_size(self, new_max_size: int) -> None:
        if new_max_size <= 0:
            return
        with self.lock:
            if new_max_size == self.max_size:
                return
            # Extract current data in logical order
            logical_data = self._get_all_unlocked()
            n_keep = min(len(logical_data), new_max_size)
            kept = logical_data[-n_keep:] if n_keep > 0 else []

            self.max_size = new_max_size
            self._positions = np.empty(new_max_size, dtype=object)
            self._policies = np.empty(new_max_size, dtype=object)
            self._values = np.zeros(new_max_size, dtype=np.float32)
            self._size = n_keep
            self._head = 0

            for i, (pos, pol, val) in enumerate(kept):
                self._positions[i] = pos
                self._policies[i] = pol
                self._values[i] = val
            self._head = n_keep % max(1, new_max_size)

    def _get_all_unlocked(self) -> List[Tuple]:
        """Return all entries as a list in logical order (oldest first)."""
        if self._size == 0:
            return []
        result = []
        for i in range(self._size):
            phys = self._logical_to_physical(i)
            result.append((self._positions[phys], self._policies[phys], self._values[phys]))
        return result

    def get_stats(self) -> dict:
        with self.lock:
            if self._size == 0:
                return {
                    "size": 0,
                    "capacity": self.max_size,
                    "fill_ratio": 0.0,
                    "value_mean": 0.0,
                    "value_std": 0.0,
                }

            values_array = self._values[:self._size] if self._size < self.max_size else self._values

            return {
                "size": self._size,
                "capacity": self.max_size,
                "fill_ratio": self._size / self.max_size,
                "value_mean": float(values_array.mean()),
                "value_std": float(values_array.std()),
            }

    def save_to_npz(self, file_path: str) -> None:
        with self.lock:
            if self._size == 0:
                return
            data = self._get_all_unlocked()
            positions = np.stack([d[0] for d in data])
            policies = np.stack([d[1] for d in data])
            values = np.array([d[2] for d in data], dtype=np.float32)
        np.savez_compressed(file_path, positions=positions, policies=policies, values=values)

    def load_from_npz(self, file_path: str) -> None:
        data = np.load(file_path)
        positions = data["positions"]
        policies = data["policies"]
        values = data["values"]

        if len(positions) == 0:
            return

        if len(positions) > self.max_size:
            positions = positions[-self.max_size:]
            policies = policies[-self.max_size:]
            values = values[-self.max_size:]

        with self.lock:
            self._size = len(positions)
            self._head = 0
            self._positions = np.empty(self.max_size, dtype=object)
            self._policies = np.empty(self.max_size, dtype=object)
            self._values = np.zeros(self.max_size, dtype=np.float32)
            for i in range(self._size):
                self._positions[i] = np.array(positions[i], dtype=np.float16)
                self._policies[i] = np.array(policies[i], dtype=np.float16)
                self._values[i] = np.float32(values[i])
            self._head = self._size % self.max_size
