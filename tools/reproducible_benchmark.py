"""Benchmark and reproducible training helper.

Two modes:
 - benchmark: run Minimax timing micro-benchmarks on a small set of FENs
 - repro-train: run a short, seeded training invocation of the individual trainer

Usage examples:
  python tools/reproducible_benchmark.py --mode benchmark --iterations 3 --max-depth 3
  python tools/reproducible_benchmark.py --mode repro-train --seed 42

This script is intentionally lightweight and safe to run on developer machines.
It sets seeds for `random`, `numpy`, and `torch` (if available) to improve reproducibility
for the short demo training run.
"""

from __future__ import annotations

import argparse
import time
import random
import sys
import os

try:
    import numpy as np
except Exception:
    np = None

try:
    import torch
except Exception:
    torch = None


FENS = [
    # Starting position
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    # King and pawn endgame
    "8/8/8/8/8/8/4k3/4K3 w - - 0 1",
    # Typical middlegame position (Sicilian-ish)
    "r1bq1rk1/pp1n1ppp/2p2n2/3p4/3P4/2N1PN2/PP3PPP/R1BQ1RK1 w - - 0 1",
    # Tactical position (open lines)
    "r4rk1/1pp1qppp/p1np1n2/4p3/2B1P3/2N2N2/PPP2PPP/R1BQR1K1 w - - 0 1",
]


def set_global_seed(seed: int):
    """Set global RNG seeds for reproducibility."""
    random.seed(seed)
    if np is not None:
        try:
            np.random.seed(int(seed))
        except Exception:
            pass
    if torch is not None:
        try:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))
            # Make CUDA determinism stronger (may slow down)
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def run_benchmark(iterations: int = 3, max_depth: int = 3):
    """Benchmark Minimax get_best_move timings for a few FENs and depths."""
    print("Running Minimax benchmark")
    try:
        from Minimax import MinimaxAI
        import chess
    except Exception as e:
        print(f"ERROR: Could not import MinimaxAI: {e}")
        return

    results = []
    for depth in range(1, max_depth + 1):
        print(f"\nDepth {depth}")
        times = []
        for it in range(iterations):
            t0 = time.perf_counter()
            for fen in FENS:
                board = chess.Board(fen)
                ai = MinimaxAI(openings={}, color=chess.WHITE, depth=depth)
                # Only measure selecting the best move once per position
                ai.get_best_move(board)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            times.append(elapsed)
            print(f"  iter {it+1}/{iterations}: {elapsed:.3f}s for {len(FENS)} positions")

        avg = sum(times) / len(times)
        results.append((depth, avg))
        print(f"  avg: {avg:.3f}s")

    print("\nBenchmark summary:")
    for depth, avg in results:
        print(f"  depth={depth}: avg_time={avg:.3f}s")


def run_repro_train(seed: int = 42):
    """Run a short, reproducible individual training invocation with seeds set.

    This will call `train.individual.main.main()` using a very small config so it
    completes quickly (Phase 1 only by default). The script sets argv before
    invoking the trainer.
    """
    print(f"Starting reproducible training run (seed={seed})")
    set_global_seed(seed)

    # Prepare arguments: run only Phase 1 for a small demo; keep checkpoints local
    args = [
        "repro_train",
        "--variant", "baseline",
        "--start-phase", "1",
        "--checkpoint-dir", "./checkpoints_repro",
        "--selfplay-workers", "1",
        "--mcts-parallel-workers", "1",
        "--max-iterations", "1",
    ]

    # Ensure train package is importable from repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        # Import the individual training entry point and call its main()
        from train.individual import main as indiv_main
    except Exception as e:
        print(f"ERROR: Could not import train.individual.main: {e}")
        print("Make sure you're running this from the repository root and dependencies are installed.")
        return

    # Replace sys.argv temporarily
    old_argv = sys.argv[:]
    sys.argv[:] = args
    try:
        import multiprocessing as mp
        mp.freeze_support()
        indiv_main.main()
    except SystemExit as e:
        # Trainer may call SystemExit on normal completion; treat as success
        print(f"Trainer exited: {e}")
    except Exception as e:
        print(f"Repro train failed: {e}")
    finally:
        sys.argv[:] = old_argv


def parse_and_run():
    parser = argparse.ArgumentParser(description="Benchmark & reproducible training helper")
    parser.add_argument("--mode", choices=["benchmark", "repro-train"], required=True,
                        help="Mode to run: benchmark (engine timing) or repro-train (seeded short training)")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations for benchmark (default 3)")
    parser.add_argument("--max-depth", type=int, default=3, help="Max search depth for benchmark (default 3)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducible training (default 42)")

    args = parser.parse_args()

    if args.mode == "benchmark":
        run_benchmark(iterations=args.iterations, max_depth=args.max_depth)
    elif args.mode == "repro-train":
        run_repro_train(seed=args.seed)


if __name__ == "__main__":
    parse_and_run()
