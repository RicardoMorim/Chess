"""Convenience wrapper to build the puzzle sidecar from the repo root.

Usage (from anywhere):
    python train/build_puzzle_sidecar.py
    python train/build_puzzle_sidecar.py --max-rows 1000
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from league.puzzle_sidecar import main

if __name__ == "__main__":
    sys.exit(main())
