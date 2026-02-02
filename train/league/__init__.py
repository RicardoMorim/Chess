"""
League Training System

A production-ready parallel self-play + MCTS training framework for chess AI.

Key components:
- Replay buffer: Persistent, per-model FIFO storage
- Self-play workers: Parallel CPU-based MCTS game generation
- League trainer: Main orchestrator (self-play → training → checkpointing → evaluation)
- Evaluator: Low-frequency regression detection
- Monitoring: Centralized metrics collection

The rule: Only MCTS self-play improves models. Everything else measures strength,
catches regressions, or reduces cold start.
"""

__version__ = "1.0.0"

from .replay_buffer import ReplayBuffer
from .league_trainer import LeagueTrainer
from .evaluator import Evaluator
from .monitoring import MetricsCollector

__all__ = [
    "ReplayBuffer",
    "LeagueTrainer",
    "Evaluator",
    "MetricsCollector",
]
