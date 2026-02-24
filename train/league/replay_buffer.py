"""
Persistent Replay Buffer for League Training
=============================================

This module implements a FIFO replay buffer that stores game data
from self-play for training. Each model variant has its own buffer
to prevent catastrophic forgetting and maintain learning stability.

Design principles:
- Per-model FIFO buffers (separate data streams)
- Capped size to prevent memory explosion
- No mixing of data between buffer and opponents (decoupling)
- Thread-safe operations for multi-process workers
"""

from collections import deque
import random
import threading
from typing import List, Tuple, Optional
import numpy as np


class ReplayBuffer:
    """
    A FIFO replay buffer that stores game trajectories.
    
    Thread-safe for concurrent adds and samples.
    Automatically discards oldest entries when max capacity reached.
    
    Each entry: (position_tensor, policy_vector, value_scalar)
    """
    
    def __init__(self, max_size: int = 200_000):
        """
        Initialize the replay buffer.
        
        Args:
            max_size: Maximum number of positions to store
        """
        self.positions = deque(maxlen=max_size)
        self.policies = deque(maxlen=max_size)
        self.values = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.max_size = max_size
    
    def add_game(self, game_trajectory: List[Tuple]) -> None:
        """
        Add a complete game trajectory to the buffer.
        
        Args:
            game_trajectory: List of (position, policy, value) tuples
                - position: numpy array, shape (channels, 8, 8)
                - policy: numpy array, shape (4672,) or similar
                - value: scalar float in [-1, 1]
        """
        with self.lock:
            for position, policy, value in game_trajectory:
                self.positions.append(position)
                self.policies.append(policy)
                self.values.append(value)
    
    def sample(self, batch_size: int) -> Tuple[List, List, List]:
        """
        Sample a random batch from the buffer.
        
        Args:
            batch_size: Number of positions to sample
        
        Returns:
            (positions, policies, values) - lists of numpy arrays
        
        Raises:
            ValueError: if batch_size > buffer size
        """
        with self.lock:
            if batch_size > len(self.positions):
                raise ValueError(
                    f"Batch size {batch_size} exceeds buffer size {len(self.positions)}"
                )
            
            # Sample indices uniformly from buffer
            indices = random.sample(range(len(self.positions)), batch_size)
            
            positions = [self.positions[i] for i in indices]
            policies = [self.policies[i] for i in indices]
            values = [self.values[i] for i in indices]
        
        return positions, policies, values
    
    def __len__(self) -> int:
        """Return current number of positions in buffer."""
        with self.lock:
            return len(self.positions)
    
    def is_ready(self, min_size: int = 256) -> bool:
        """
        Check if buffer has enough data for training.
        
        Args:
            min_size: Minimum buffer size required
        
        Returns:
            True if len(buffer) >= min_size
        """
        return len(self) >= min_size
    
    def clear(self) -> None:
        """Clear all data from the buffer."""
        with self.lock:
            self.positions.clear()
            self.policies.clear()
            self.values.clear()

    def set_max_size(self, new_max_size: int) -> None:
        """Resize the buffer capacity, keeping the most recent entries."""
        if new_max_size <= 0:
            return
        with self.lock:
            if new_max_size == self.max_size:
                return
            self.positions = deque(list(self.positions)[-new_max_size:], maxlen=new_max_size)
            self.policies = deque(list(self.policies)[-new_max_size:], maxlen=new_max_size)
            self.values = deque(list(self.values)[-new_max_size:], maxlen=new_max_size)
            self.max_size = new_max_size
    
    def get_stats(self) -> dict:
        """
        Return buffer statistics for monitoring.
        
        Returns:
            dict with keys: size, capacity, fill_ratio, value_mean, value_std
        """
        with self.lock:
            size = len(self.positions)
            
            if size == 0:
                return {
                    "size": 0,
                    "capacity": self.max_size,
                    "fill_ratio": 0.0,
                    "value_mean": 0.0,
                    "value_std": 0.0,
                }
            
            values_array = np.array(list(self.values))
            
            return {
                "size": size,
                "capacity": self.max_size,
                "fill_ratio": size / self.max_size,
                "value_mean": float(values_array.mean()),
                "value_std": float(values_array.std()),
            }

    def save_to_npz(self, file_path: str) -> None:
        """Persist the buffer to a compressed .npz file."""
        with self.lock:
            size = len(self.positions)
            if size == 0:
                return
            positions = np.stack(list(self.positions))
            policies = np.stack(list(self.policies))
            values = np.array(list(self.values), dtype=np.float32)
        np.savez_compressed(file_path, positions=positions, policies=policies, values=values)

    def load_from_npz(self, file_path: str) -> None:
        """Load the buffer from a .npz file, keeping the most recent entries if oversized."""
        data = np.load(file_path)
        positions = data["positions"]
        policies = data["policies"]
        values = data["values"]

        if len(positions) == 0:
            return

        # Keep the most recent samples if file exceeds max_size
        if len(positions) > self.max_size:
            positions = positions[-self.max_size:]
            policies = policies[-self.max_size:]
            values = values[-self.max_size:]

        with self.lock:
            self.positions = deque(list(positions), maxlen=self.max_size)
            self.policies = deque(list(policies), maxlen=self.max_size)
            self.values = deque(list(values), maxlen=self.max_size)
