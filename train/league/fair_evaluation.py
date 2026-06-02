"""Fair evaluation helpers for league training.

This module defines a shared opening suite and small utility helpers used to
compare the three league models under identical starting conditions.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import chess


FAIR_OPENING_LINES: Sequence[Tuple[str, ...]] = (
    ("e2e4", "e7e5", "g1f3", "b8c6"),
    ("d2d4", "d7d5", "c2c4", "e7e6"),
    ("c2c4", "e7e5", "g1f3", "b8c6"),
    ("g1f3", "d7d5", "c2c4", "e7e6"),
    ("e2e4", "c7c5", "g1f3", "d7d6"),
    ("d2d4", "g8f6", "c2c4", "g7g6"),
    ("e2e4", "e7e6", "d2d4", "d7d5"),
    ("g2g3", "d7d5", "f1g2", "e7e5"),
    ("b2b3", "e7e5", "c1b2", "b8c6"),
    ("d2d4", "g8f6", "c1g5", "d7d5"),
    ("e2e4", "c7c6", "d2d4", "d7d5"),
    ("c2c4", "g8f6", "d2d4", "g7g6"),
)


def build_fair_opening_fens() -> List[str]:
    """Build the shared opening suite used by all model comparisons.

    The returned FENs are deterministic and position-balanced so each model is
    evaluated on the exact same starting positions.
    """
    fens: List[str] = []
    seen = set()

    for line in FAIR_OPENING_LINES:
        board = chess.Board()
        valid = True
        try:
            for move_uci in line:
                move = chess.Move.from_uci(move_uci)
                if move not in board.legal_moves:
                    valid = False
                    break
                board.push(move)
        except Exception:
            valid = False

        if valid:
            fen = board.fen()
            if fen not in seen:
                fens.append(fen)
                seen.add(fen)

    # Always include the starting position as a universal baseline.
    start_fen = chess.Board().fen()
    if start_fen not in seen:
        fens.insert(0, start_fen)
        seen.add(start_fen)

    return fens


def round_robin_pairs(variants: Sequence[str]) -> List[Tuple[str, str]]:
    """Return all unique round-robin pairings for a set of variants."""
    return list(combinations(variants, 2))
