"""
Persistent Replay Buffer for League Training
=============================================

This module implements a FIFO replay buffer that stores game data
from self-play for training. Each model variant has its own buffer
to prevent catastrophic forgetting and maintain learning stability.

Design principles:
- Per-model FIFO buffers (separate data streams)
- Capped size to prevent memory explosion
- Pre-allocated flat numpy arrays (no per-element Python objects)
- Thread-safe operations for multi-process workers

Memory layout (compact form, default):
  positions  : (max_size, POS_CHANNELS, 8, 8)  float16
  policies   : (max_size, POLICY_SIZE)         float16
  values     : (max_size,)                     float32
  ~10.5 KB per entry (vs ~26 KB with the old object-dtype version)

The compact layout makes large buffers feasible on memory-constrained
laptops (~2GB per 100K-entry buffer at 22 channels).
"""

from __future__ import annotations

import logging
import os
import random
import threading
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# Defaults — overridable per-instance. 22 channels matches the
# "attack" / "est" variants; "baseline" uses 18. We use 22 to avoid
# channel-mismatch errors when variants share buffers.
DEFAULT_POS_CHANNELS = 22
# 73 planes x 64 squares = 4672 (AlphaZero move encoding used by the
# model + MCTS). Was 4096 (64 from * 64 to) which silently truncated
# MCTS policies and broke training.
DEFAULT_POLICY_SIZE = 4672


class ReplayBuffer:
    """
    A FIFO replay buffer backed by pre-allocated flat numpy arrays.

    Thread-safe for concurrent adds and samples.
    Automatically overwrites oldest entries when max capacity is reached.

    Each entry: (position_tensor, policy_vector, value_scalar)

    All operations are O(1) except sample() which is O(batch_size) and
    get_stats() which is O(size).
    """

    def __init__(
        self,
        max_size: int = 200_000,
        pos_channels: int = DEFAULT_POS_CHANNELS,
        policy_size: int = DEFAULT_POLICY_SIZE,
        pos_dtype: type = np.float16,
        policy_dtype: type = np.float16,
    ):
        self.max_size = int(max_size)
        self.pos_channels = int(pos_channels)
        self.policy_size = int(policy_size)
        self.pos_dtype = pos_dtype
        self.policy_dtype = policy_dtype
        self.lock = threading.Lock()
        # Pre-allocated flat arrays (zero Python-object overhead).
        self._positions = np.zeros(
            (self.max_size, self.pos_channels, 8, 8), dtype=self.pos_dtype
        )
        self._policies = np.zeros(
            (self.max_size, self.policy_size), dtype=self.policy_dtype
        )
        self._values = np.zeros(self.max_size, dtype=np.float32)
        # Validity bit — 1.0 if the slot holds real data, 0.0 if it's
        # a never-written slot that should be ignored when sampling.
        self._valid = np.zeros(self.max_size, dtype=np.bool_)
        self._size = 0
        self._head = 0
        # Stats (best-effort, for get_stats without full scans)
        self._value_sum = 0.0
        self._value_sumsq = 0.0

    # ---- internals ----------------------------------------------------------

    def _add_one(self, position, policy, value: float) -> None:
        # If we're overwriting an existing slot, remove its contribution
        # from the running stats.
        if self._valid[self._head]:
            old_v = float(self._values[self._head])
            self._value_sum -= old_v
            self._value_sumsq -= old_v * old_v
        # Compact copy — accepts (C,8,8) array, list, or anything
        # convertible. dtype conversion happens here.
        pos_arr = np.asarray(position, dtype=self.pos_dtype)
        if pos_arr.shape != (self.pos_channels, 8, 8):
            # Truncate or pad along the channel axis if a smaller/larger
            # tensor is provided (e.g. 18-channel baseline into a
            # 22-channel buffer).
            flat = pos_arr.reshape(-1, 8, 8)
            if flat.shape[0] >= self.pos_channels:
                pos_arr = flat[: self.pos_channels]
            else:
                pad = np.zeros(
                    (self.pos_channels - flat.shape[0], 8, 8), dtype=self.pos_dtype
                )
                pos_arr = np.concatenate([flat, pad], axis=0)
        pol_arr = np.asarray(policy, dtype=self.policy_dtype)
        if pol_arr.shape[0] >= self.policy_size:
            pol_arr = pol_arr[: self.policy_size]
        else:
            padded = np.zeros(self.policy_size, dtype=self.policy_dtype)
            padded[: pol_arr.shape[0]] = pol_arr
            pol_arr = padded
        self._positions[self._head] = pos_arr
        self._policies[self._head] = pol_arr
        self._values[self._head] = np.float32(value)
        self._valid[self._head] = True
        self._value_sum += float(value)
        self._value_sumsq += float(value) * float(value)
        # Advance head
        self._head = (self._head + 1) % self.max_size
        if self._size < self.max_size:
            self._size += 1

    # ---- public API ---------------------------------------------------------

    def add_game(self, game_trajectory: List[Tuple]) -> None:
        """Append an entire game's worth of (pos, policy, value) entries."""
        with self.lock:
            for position, policy, value in game_trajectory:
                self._add_one(position, policy, value)

    def add_many(self, positions, policies, values) -> None:
        """Vectorized add — bulk-copy a batch of entries without the loop.

        All three inputs must have matching leading length. Each ``position``
        is shape (C,8,8); each ``policy`` is shape (POLICY_SIZE,); ``values``
        is a 1D array-like.
        """
        n = len(positions)
        if n == 0:
            return
        if len(policies) != n or len(values) != n:
            raise ValueError("positions/policies/values length mismatch")
        with self.lock:
            pos_arr = np.asarray(positions, dtype=self.pos_dtype).reshape(n, self.pos_channels, 8, 8)
            pol_arr = np.asarray(policies, dtype=self.policy_dtype).reshape(n, self.policy_size)
            val_arr = np.asarray(values, dtype=np.float32).reshape(n)
            # Write slot-by-slot so we honour the circular head.
            for i in range(n):
                self._positions[self._head] = pos_arr[i]
                self._policies[self._head] = pol_arr[i]
                self._values[self._head] = val_arr[i]
                self._valid[self._head] = True
                self._head = (self._head + 1) % self.max_size
            self._size = min(self.max_size, self._size + n)

    def sample(self, batch_size: int, return_numpy: bool = False):
        """Sample a random batch.

        Args:
            batch_size: Number of entries to sample.
            return_numpy: If True, return a single (positions, policies, values)
                triple of numpy arrays (fast path for the training loop).
                If False, return the legacy list-of-arrays API.

        Returns:
            (positions, policies, values) where each element is either a
            numpy array (return_numpy=True) or a list of numpy arrays
            (return_numpy=False).
        """
        with self.lock:
            if batch_size > self._size:
                raise ValueError(
                    f"Batch size {batch_size} exceeds buffer size {self._size}"
                )
            logical_indices = random.sample(range(self._size), batch_size)
            # Map logical -> physical
            if self._size < self.max_size:
                physical = list(logical_indices)
            else:
                physical = [(self._head + i) % self.max_size for i in logical_indices]
            if return_numpy:
                positions = self._positions[physical].astype(np.float32, copy=False)
                policies = self._policies[physical].astype(np.float32, copy=False)
                values = self._values[physical].copy()
                return positions, policies, values
            # Legacy list-of-arrays form
            positions = [self._positions[i].astype(np.float32, copy=True) for i in physical]
            policies = [self._policies[i].astype(np.float32, copy=True) for i in physical]
            values = [float(self._values[i]) for i in physical]
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
            self._valid.fill(False)
            self._value_sum = 0.0
            self._value_sumsq = 0.0

    def set_max_size(self, new_max_size: int) -> None:
        """Shrink or grow the buffer live.

        If ``new_max_size`` is smaller than the current size, the most
        recent entries are kept. If larger, additional slots are zero-
        filled and ignored until written. No data is copied unless a
        shrink is required.
        """
        if new_max_size <= 0:
            return
        new_max_size = int(new_max_size)
        with self.lock:
            if new_max_size == self.max_size:
                return
            if new_max_size < self.max_size:
                # Shrink: extract current data in chronological order,
                # keep the last ``new_max_size`` entries, then re-pack
                # into a fresh pre-allocated array.
                n_keep = min(self._size, new_max_size)
                if n_keep == 0:
                    self._positions = np.zeros(
                        (new_max_size, self.pos_channels, 8, 8), dtype=self.pos_dtype
                    )
                    self._policies = np.zeros(
                        (new_max_size, self.policy_size), dtype=self.policy_dtype
                    )
                    self._values = np.zeros(new_max_size, dtype=np.float32)
                    self._valid = np.zeros(new_max_size, dtype=np.bool_)
                    self.max_size = new_max_size
                    self._size = 0
                    self._head = 0
                    self._value_sum = 0.0
                    self._value_sumsq = 0.0
                    return
                # Logical->physical mapping
                if self._size < self.max_size:
                    src = list(range(self._size))
                else:
                    src = [(self._head + i) % self.max_size for i in range(self._size)]
                src = src[-n_keep:]
                new_pos = np.zeros(
                    (new_max_size, self.pos_channels, 8, 8), dtype=self.pos_dtype
                )
                new_pol = np.zeros(
                    (new_max_size, self.policy_size), dtype=self.policy_dtype
                )
                new_val = np.zeros(new_max_size, dtype=np.float32)
                new_valid = np.zeros(new_max_size, dtype=np.bool_)
                for i, phys in enumerate(src):
                    new_pos[i] = self._positions[phys]
                    new_pol[i] = self._policies[phys]
                    new_val[i] = self._values[phys]
                    new_valid[i] = True
                self._positions = new_pos
                self._policies = new_pol
                self._values = new_val
                self._valid = new_valid
                self.max_size = new_max_size
                self._size = n_keep
                self._head = n_keep % new_max_size
                # Recompute stats from scratch (cheap; we have them all)
                self._value_sum = float(self._values[: n_keep].sum())
                self._value_sumsq = float((self._values[: n_keep] ** 2).sum())
            else:
                # Grow: allocate bigger arrays and copy.
                new_pos = np.zeros(
                    (new_max_size, self.pos_channels, 8, 8), dtype=self.pos_dtype
                )
                new_pol = np.zeros(
                    (new_max_size, self.policy_size), dtype=self.policy_dtype
                )
                new_val = np.zeros(new_max_size, dtype=np.float32)
                new_valid = np.zeros(new_max_size, dtype=np.bool_)
                if self._size > 0:
                    if self._size < self.max_size:
                        src = list(range(self._size))
                    else:
                        src = [(self._head + i) % self.max_size for i in range(self._size)]
                    for i, phys in enumerate(src):
                        new_pos[i] = self._positions[phys]
                        new_pol[i] = self._policies[phys]
                        new_val[i] = self._values[phys]
                        new_valid[i] = True
                self._positions = new_pos
                self._policies = new_pol
                self._values = new_val
                self._valid = new_valid
                self.max_size = new_max_size
                self._head = self._size % new_max_size

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
            # Use the running stats for O(1) mean/std (avoids full scan)
            n = self._size
            mean = self._value_sum / n
            var = max(0.0, self._value_sumsq / n - mean * mean)
            return {
                "size": self._size,
                "capacity": self.max_size,
                "fill_ratio": self._size / self.max_size,
                "value_mean": float(mean),
                "value_std": float(var ** 0.5),
            }

    def save_to_npz(self, file_path: str) -> None:
        with self.lock:
            if self._size == 0:
                return
            # Return data in logical order
            if self._size < self.max_size:
                src = list(range(self._size))
            else:
                src = [(self._head + i) % self.max_size for i in range(self._size)]
            positions = self._positions[src].astype(np.float16, copy=False)
            policies = self._policies[src].astype(np.float16, copy=False)
            values = self._values[src].copy()
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
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
            self._positions = np.zeros(
                (self.max_size, self.pos_channels, 8, 8), dtype=self.pos_dtype
            )
            self._policies = np.zeros(
                (self.max_size, self.policy_size), dtype=self.policy_dtype
            )
            self._values = np.zeros(self.max_size, dtype=np.float32)
            self._valid = np.zeros(self.max_size, dtype=np.bool_)
            # Normalize shape to the buffer's expected channel count
            for i in range(self._size):
                p = positions[i]
                if p.shape[0] != self.pos_channels:
                    if p.shape[0] >= self.pos_channels:
                        p = p[: self.pos_channels]
                    else:
                        pad = np.zeros(
                            (self.pos_channels - p.shape[0], 8, 8), dtype=self.pos_dtype
                        )
                        p = np.concatenate([p, pad], axis=0)
                pol = policies[i]
                if pol.shape[0] < self.policy_size:
                    pad = np.zeros(self.policy_size - pol.shape[0], dtype=self.policy_dtype)
                    pol = np.concatenate([pol, pad], axis=0)
                elif pol.shape[0] > self.policy_size:
                    pol = pol[: self.policy_size]
                self._positions[i] = p.astype(self.pos_dtype, copy=False)
                self._policies[i] = pol.astype(self.policy_dtype, copy=False)
                self._values[i] = np.float32(values[i])
                self._valid[i] = True
            self._value_sum = float(self._values[: self._size].sum())
            self._value_sumsq = float((self._values[: self._size] ** 2).sum())
            self._head = self._size % self.max_size
