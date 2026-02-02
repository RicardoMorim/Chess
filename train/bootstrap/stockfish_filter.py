"""
Stockfish Puzzle Filter
=======================

Filter and validate puzzle quality using Stockfish engine.

This is a DATA PREPARATION tool - it doesn't train models.
Use it to:
- Verify puzzle solutions are correct
- Filter out ambiguous puzzles
- Add difficulty ratings
"""

import sys
import chess
import chess.engine
from pathlib import Path
import logging
import csv
import json
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class StockfishFilter:
    """
    Filter puzzles using Stockfish analysis.
    
    Validates that:
    1. The solution move is indeed the best move
    2. The evaluation difference is significant
    3. No alternative moves are equally good
    """
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        threads: int = 4,
        hash_mb: int = 512,
        depth: int = 20,
    ):
        """
        Initialize Stockfish filter.
        
        Args:
            stockfish_path: Path to Stockfish executable
            threads: Number of threads for Stockfish
            hash_mb: Hash table size in MB
            depth: Analysis depth
        """
        self.stockfish_path = stockfish_path
        self.threads = threads
        self.hash_mb = hash_mb
        self.depth = depth
        
        self.engine = None
    
    def _get_engine(self) -> chess.engine.SimpleEngine:
        """Get or create Stockfish engine."""
        if self.engine is None:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
                self.engine.configure({
                    "Threads": self.threads,
                    "Hash": self.hash_mb,
                })
            except Exception as e:
                logger.error(f"Failed to start Stockfish: {e}")
                raise
        return self.engine
    
    def close(self):
        """Close Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None
    
    def validate_puzzle(
        self,
        fen: str,
        solution_move: str,
        min_eval_diff: int = 100,  # centipawns
    ) -> Dict[str, Any]:
        """
        Validate a single puzzle.
        
        Args:
            fen: Board position FEN
            solution_move: Expected solution move (UCI format)
            min_eval_diff: Minimum evaluation difference for valid puzzle
        
        Returns:
            {
                "valid": bool,
                "reason": str,
                "best_move": str,
                "eval": float,
                "second_best_eval": float,
            }
        """
        
        engine = self._get_engine()
        board = chess.Board(fen)
        
        try:
            solution = chess.Move.from_uci(solution_move)
            
            if solution not in board.legal_moves:
                return {
                    "valid": False,
                    "reason": "Solution move is illegal",
                    "best_move": None,
                    "eval": None,
                }
            
            # Analyze position
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=self.depth),
                multipv=3,  # Get top 3 moves
            )
            
            # Handle single PV result
            if isinstance(info, dict):
                info = [info]
            
            if not info:
                return {
                    "valid": False,
                    "reason": "No analysis result",
                    "best_move": None,
                    "eval": None,
                }
            
            # Get best move and eval
            best_info = info[0]
            best_move = best_info["pv"][0] if best_info.get("pv") else None
            best_score = best_info.get("score")
            
            if best_move is None:
                return {
                    "valid": False,
                    "reason": "No best move found",
                    "best_move": None,
                    "eval": None,
                }
            
            # Convert score to centipawns
            if best_score:
                if best_score.is_mate():
                    best_eval = 10000 * (1 if best_score.mate() > 0 else -1)
                else:
                    best_eval = best_score.relative.score()
            else:
                best_eval = 0
            
            # Get second best eval
            second_eval = None
            if len(info) > 1:
                second_score = info[1].get("score")
                if second_score:
                    if second_score.is_mate():
                        second_eval = 10000 * (1 if second_score.mate() > 0 else -1)
                    else:
                        second_eval = second_score.relative.score()
            
            # Validate
            if best_move != solution:
                return {
                    "valid": False,
                    "reason": f"Solution {solution_move} is not best move ({best_move.uci()})",
                    "best_move": best_move.uci(),
                    "eval": best_eval,
                    "second_best_eval": second_eval,
                }
            
            # Check if puzzle is clear (big eval difference)
            if second_eval is not None:
                eval_diff = abs(best_eval - second_eval) if second_eval else abs(best_eval)
                if eval_diff < min_eval_diff:
                    return {
                        "valid": False,
                        "reason": f"Eval difference too small ({eval_diff} cp)",
                        "best_move": best_move.uci(),
                        "eval": best_eval,
                        "second_best_eval": second_eval,
                    }
            
            return {
                "valid": True,
                "reason": "Puzzle validated",
                "best_move": best_move.uci(),
                "eval": best_eval,
                "second_best_eval": second_eval,
            }
        
        except Exception as e:
            return {
                "valid": False,
                "reason": f"Error: {str(e)}",
                "best_move": None,
                "eval": None,
            }
    
    def filter_puzzle_file(
        self,
        input_file: str,
        output_file: str,
        min_eval_diff: int = 100,
        max_puzzles: Optional[int] = None,
    ) -> Dict[str, int]:
        """
        Filter puzzles from input file to output file.
        
        Args:
            input_file: Input puzzle file (CSV/JSON)
            output_file: Output file for valid puzzles
            min_eval_diff: Minimum eval difference
            max_puzzles: Maximum puzzles to process
        
        Returns:
            {"total": N, "valid": M, "invalid": K}
        """
        
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        puzzles = []
        
        # Load puzzles
        if input_path.suffix == '.csv':
            with open(input_path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_puzzles and i >= max_puzzles:
                        break
                    puzzles.append(row)
        elif input_path.suffix == '.json':
            with open(input_path, 'r') as f:
                puzzles = json.load(f)[:max_puzzles] if max_puzzles else json.load(f)
        
        logger.info(f"Loaded {len(puzzles)} puzzles from {input_file}")
        
        # Validate puzzles
        valid_puzzles = []
        stats = {"total": len(puzzles), "valid": 0, "invalid": 0}
        
        for i, puzzle in enumerate(puzzles):
            fen = puzzle.get('FEN', puzzle.get('fen', ''))
            moves = puzzle.get('Moves', puzzle.get('moves', '')).split()
            
            if not fen or not moves:
                stats["invalid"] += 1
                continue
            
            # Get solution move
            board = chess.Board(fen)
            if len(moves) >= 2:
                try:
                    board.push_uci(moves[0])
                    solution = moves[1]
                except:
                    stats["invalid"] += 1
                    continue
            else:
                solution = moves[0]
            
            # Validate
            result = self.validate_puzzle(
                board.fen(),
                solution,
                min_eval_diff=min_eval_diff,
            )
            
            if result["valid"]:
                puzzle["validated"] = True
                puzzle["stockfish_eval"] = result["eval"]
                valid_puzzles.append(puzzle)
                stats["valid"] += 1
            else:
                stats["invalid"] += 1
                logger.debug(f"Invalid puzzle {i}: {result['reason']}")
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i+1}/{len(puzzles)}: {stats['valid']} valid")
        
        # Save valid puzzles
        if output_path.suffix == '.csv':
            if valid_puzzles:
                with open(output_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=valid_puzzles[0].keys())
                    writer.writeheader()
                    writer.writerows(valid_puzzles)
        elif output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(valid_puzzles, f, indent=2)
        
        logger.info(f"Saved {stats['valid']} valid puzzles to {output_file}")
        logger.info(f"Stats: {stats}")
        
        return stats
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def add_stockfish_eval(
    puzzle_file: str,
    output_file: str,
    stockfish_path: str = "stockfish",
    depth: int = 20,
) -> None:
    """
    Add Stockfish evaluation to puzzle file.
    
    Args:
        puzzle_file: Input puzzle file
        output_file: Output file with evaluations
        stockfish_path: Path to Stockfish
        depth: Analysis depth
    """
    
    with StockfishFilter(stockfish_path=stockfish_path, depth=depth) as sf:
        sf.filter_puzzle_file(puzzle_file, output_file)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter puzzles with Stockfish")
    parser.add_argument("--input", required=True, help="Input puzzle file")
    parser.add_argument("--output", required=True, help="Output file")
    parser.add_argument("--stockfish", default="stockfish", help="Stockfish path")
    parser.add_argument("--depth", type=int, default=20, help="Analysis depth")
    parser.add_argument("--min-eval-diff", type=int, default=100, help="Min eval diff (cp)")
    parser.add_argument("--max-puzzles", type=int, default=None, help="Max puzzles")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    with StockfishFilter(
        stockfish_path=args.stockfish,
        depth=args.depth,
    ) as sf:
        stats = sf.filter_puzzle_file(
            args.input,
            args.output,
            min_eval_diff=args.min_eval_diff,
            max_puzzles=args.max_puzzles,
        )
    
    print(f"\nDone. Valid: {stats['valid']}/{stats['total']}")
