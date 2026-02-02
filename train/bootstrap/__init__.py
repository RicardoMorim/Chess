"""
Bootstrap Training Module
=========================

One-time supervised learning on puzzles to reduce cold-start.
This is NOT part of the main league training loop.

After bootstrap, only MCTS self-play improves models.

Components:
- puzzle_train: Train models on tactical puzzles
- puzzle_eval: Evaluate puzzle-solving accuracy
- stockfish_filter: Filter/validate puzzle quality
"""

__all__ = [
    "puzzle_train",
    "puzzle_eval", 
    "stockfish_filter",
]
