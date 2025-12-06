"""
Test script for parallel Minimax search.
Compares single-threaded vs multi-process (Lazy SMP) performance.
"""

import time
import chess
from Minimax_improved import MinimaxAI

def test_parallel_search():
    """Test parallel vs single-threaded search."""
    print("=" * 60)
    print("MINIMAX PARALLEL SEARCH TEST")
    print("=" * 60)
    
    # Test position (Italian Game)
    board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
    for m in moves:
        board.push(chess.Move.from_uci(m))
    
    print(f"\nTest position: {board.fen()}")
    print(board)
    print()
    
    # Test single-threaded
    print("-" * 60)
    print("SINGLE-THREADED SEARCH (depth 5)")
    print("-" * 60)
    
    engine_single = MinimaxAI({}, 'w', depth=5, use_parallel=False)
    
    start = time.time()
    move_single = engine_single.get_best_move(board)
    time_single = time.time() - start
    
    print(f"\nBest move: {move_single}")
    print(f"Time: {time_single:.2f}s")
    print(f"Nodes: {engine_single.nodes_searched:,}")
    print(f"NPS: {engine_single.nodes_searched / time_single:,.0f}")
    
    # Test parallel
    print("\n" + "-" * 60)
    print("PARALLEL SEARCH - LAZY SMP (depth 5)")
    print("-" * 60)
    
    engine_parallel = MinimaxAI({}, 'w', depth=5, use_parallel=True)
    print(f"Using {engine_parallel.num_workers} worker processes")
    
    start = time.time()
    move_parallel = engine_parallel.get_best_move(board)
    time_parallel = time.time() - start
    
    print(f"\nBest move: {move_parallel}")
    print(f"Time: {time_parallel:.2f}s")
    print(f"Nodes: {engine_parallel.nodes_searched:,}")
    print(f"NPS: {engine_parallel.nodes_searched / time_parallel:,.0f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Single-threaded: {time_single:.2f}s")
    print(f"Parallel ({engine_parallel.num_workers} workers): {time_parallel:.2f}s")
    
    if time_parallel < time_single:
        speedup = time_single / time_parallel
        print(f"Speedup: {speedup:.2f}x faster with parallel search!")
    else:
        print("Note: For short searches, process overhead may exceed gains.")
        print("Parallel search shines at deeper depths (6+).")


if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    test_parallel_search()
