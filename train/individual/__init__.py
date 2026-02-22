"""
Individual Training Module
==========================

Single-threaded 3-phase curriculum training for chess AI.
Uses core/ modules for shared functionality.

Phases:
1. Puzzle Bootcamp - Bootstrap tactical knowledge
2. Transition - Blend puzzles with initial self-play
3. Pure Self-Play - Convergence loop

Usage:
    python -m individual.main --variant baseline
"""

import sys
from pathlib import Path

# Allow `import individual` / `python -m train.individual.main` from repo root.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from .curriculum import phase1_puzzle_bootcamp, phase2_transition, phase3_pure_selfplay
from .checkmate import run_checkmate_bootcamp, run_checkmate_reinforcement

__all__ = [
    'phase1_puzzle_bootcamp',
    'phase2_transition', 
    'phase3_pure_selfplay',
    'run_checkmate_bootcamp',
    'run_checkmate_reinforcement',
]
