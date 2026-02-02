"""
Stockfish Evaluation Utilities (NOT for training)
==================================================

This module provides Stockfish-based evaluation tools for:
- Puzzle filtering (quality gate)
- Model strength benchmarking
- Regression detection during training

IMPORTANT: These utilities are for EVALUATION ONLY.
DO NOT use for policy or value training targets.
"""
import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from pathlib import Path
import os


# =============================================================================
# STOCKFISH EVALUATOR
# =============================================================================

class StockfishEvaluator:
    """Stockfish evaluator for benchmarking and puzzle filtering."""
    
    def __init__(self, stockfish_path: str = None, depth: int = 18, threads: int = 1, hash_mb: int = 128):
        """
        Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish executable (auto-detected if None)
            depth: Search depth for evaluations (default: 18 for frozen eval)
            threads: Number of CPU threads for Stockfish (default: 1 for determinism)
            hash_mb: Hash table size in MB (default: 128)
        """
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine = None
        
        # Auto-detect Stockfish path
        if stockfish_path is None:
            stockfish_path = self._find_stockfish()
        
        self.stockfish_path = stockfish_path
        
        if stockfish_path and Path(stockfish_path).exists():
            self._init_engine()
    
    def _find_stockfish(self) -> Optional[str]:
        """Try to find Stockfish executable."""
        common_paths = [
            "./stockfish/stockfish.exe",
            "./stockfish/stockfish-windows-x86-64-avx2.exe",
            "../stockfish/stockfish.exe",
            "C:/stockfish/stockfish.exe",
        ]
        for path in common_paths:
            if Path(path).exists():
                return str(Path(path).resolve())
        return None
    
    def _init_engine(self):
        """Initialize Stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb,
            })
            print(f"✓ Stockfish initialized: {self.stockfish_path} (depth={self.depth}, threads={self.threads}, hash={self.hash_mb}MB)")
        except Exception as e:
            print(f"⚠ Could not initialize Stockfish: {e}")
            self.engine = None
    
    def evaluate(self, board: chess.Board, from_side_to_move: bool = True) -> float:
        """
        Evaluate a position using Stockfish.
        
        Args:
            board: Chess position to evaluate
            from_side_to_move: If True, return eval from perspective of side to move
        
        Returns:
            Evaluation in centipawns (positive = side to move advantage)
            Returns +/- 10000 for mate
        """
        if self.engine is None:
            raise RuntimeError("Stockfish engine not initialized")
        
        result = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score = result['score'].pov(board.turn if from_side_to_move else chess.WHITE)
        
        if score.is_mate():
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000
        else:
            return score.score(mate_score=10000)
    
    def close(self):
        """Close the Stockfish engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None


# =============================================================================
# PUZZLE FILTERING
# =============================================================================

def filter_puzzles_by_stockfish(
    puzzles: List[Tuple],
    evaluator: StockfishEvaluator,
    min_eval: float = 300,  # +3.0 in centipawns
    verbose: bool = True
) -> List[Tuple]:
    """
    Filter puzzles to only include positions where side-to-move has clear advantage.
    
    Args:
        puzzles: List of puzzle tuples (fen, solution, ...)
        evaluator: StockfishEvaluator instance
        min_eval: Minimum evaluation (in centipawns) from side-to-move perspective
        verbose: Print progress
    
    Returns:
        Filtered list of puzzles
    """
    if evaluator.engine is None:
        print("⚠ Stockfish not available, returning all puzzles")
        return puzzles
    
    filtered = []
    for i, puzzle in enumerate(puzzles):
        try:
            fen = puzzle[0] if isinstance(puzzle[0], str) else puzzle[0]
            board = chess.Board(fen)
            
            eval_cp = evaluator.evaluate(board, from_side_to_move=True)
            
            if eval_cp >= min_eval:
                filtered.append(puzzle)
        except Exception as e:
            if verbose and i < 10:
                print(f"  Error evaluating puzzle {i}: {e}")
        
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Filtered {i + 1}/{len(puzzles)} puzzles, kept {len(filtered)}")
    
    if verbose:
        print(f"Puzzle filtering complete: {len(filtered)}/{len(puzzles)} passed (min_eval={min_eval}cp)")
    
    return filtered


# =============================================================================
# MODEL BENCHMARKING
# =============================================================================

def evaluate_model_vs_stockfish(
    model,
    device,
    evaluator: StockfishEvaluator,
    num_positions: int = 150,  # Frozen eval protocol: 150 positions
    input_channels: int = 22
) -> Dict[str, float]:
    """
    Benchmark model against Stockfish on random positions.
    
    Args:
        model: Neural network model
        device: Computation device
        evaluator: StockfishEvaluator instance
        num_positions: Number of positions to evaluate
        input_channels: Model input channel count
    
    Returns:
        Dictionary with benchmark metrics
    """
    from data import board_to_tensor
    
    if evaluator.engine is None:
        return {"error": "Stockfish not available"}
    
    model.eval()
    
    # Generate random mid-game positions
    positions = _generate_benchmark_positions(num_positions)
    
    value_errors = []
    policy_agreements = []
    
    for board in positions:
        try:
            # Stockfish evaluation
            sf_eval = evaluator.evaluate(board, from_side_to_move=True)
            sf_eval_normalized = np.tanh(sf_eval / 400)  # Normalize to [-1, 1]
            
            # Get Stockfish best move
            result = evaluator.engine.play(board, chess.engine.Limit(depth=evaluator.depth))
            sf_move = result.move
            
            # Model evaluation
            tensor = torch.tensor(
                board_to_tensor(board, board.fullmove_number, input_channels),
                dtype=torch.float32
            ).unsqueeze(0).to(device)
            
            with torch.no_grad():
                policy_logits, value_pred = model(tensor)
            
            model_value = value_pred.item()
            policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
            
            # Value error
            value_errors.append(abs(model_value - sf_eval_normalized))
            
            # Policy agreement (does model's top move match Stockfish?)
            from data import get_move_index
            legal_moves = list(board.legal_moves)
            if legal_moves:
                move_probs = []
                for move in legal_moves:
                    idx = get_move_index(move)
                    move_probs.append((policy[idx] if idx < len(policy) else 0, move))
                
                model_best = max(move_probs, key=lambda x: x[0])[1]
                policy_agreements.append(1.0 if model_best == sf_move else 0.0)
        
        except Exception as e:
            continue
    
    model.train()
    
    return {
        "value_mae": np.mean(value_errors) if value_errors else float('nan'),
        "policy_agreement": np.mean(policy_agreements) if policy_agreements else float('nan'),
        "positions_evaluated": len(value_errors),
    }


def load_frozen_positions(filepath: str = None) -> List[chess.Board]:
    """Load frozen evaluation positions from file.
    
    Args:
        filepath: Path to FEN file (default: eval_positions_v1.txt)
    
    Returns:
        List of chess.Board objects
    """
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "eval_positions_v1.txt")
    
    positions = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    try:
                        board = chess.Board(line)
                        positions.append(board)
                    except ValueError:
                        continue
    
    return positions


def sample_pro_game_positions(
    num_positions: int = 150,
    pgn_dir: str = None,
    min_move: int = 15,
    max_move: int = 50,
    seed: int = None
) -> List[chess.Board]:
    """Sample random middlegame positions from pro player games.
    
    This avoids any potential overfitting to fixed evaluation positions
    by sampling fresh positions from real games each time.
    
    Args:
        num_positions: Number of positions to sample
        pgn_dir: Directory containing PGN files (default: chess_pgns/pros)
        min_move: Minimum move number (to skip openings)
        max_move: Maximum move number (to focus on middlegame)
        seed: Random seed for reproducibility within a run
    
    Returns:
        List of chess.Board objects
    """
    import chess.pgn
    import random
    import glob
    
    if seed is not None:
        random.seed(seed)
    
    if pgn_dir is None:
        pgn_dir = os.path.join(os.path.dirname(__file__), "chess_pgns", "pros")
    
    # Get all PGN files
    pgn_files = glob.glob(os.path.join(pgn_dir, "*.pgn"))
    if not pgn_files:
        print(f"⚠ No PGN files found in {pgn_dir}")
        return []
    
    positions = []
    random.shuffle(pgn_files)
    
    for pgn_file in pgn_files:
        if len(positions) >= num_positions:
            break
        
        try:
            with open(pgn_file, 'r', errors='ignore') as f:
                while len(positions) < num_positions:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Get the move list
                    board = game.board()
                    moves = list(game.mainline_moves())
                    
                    # Skip short games
                    if len(moves) < max_move:
                        continue
                    
                    # Pick a random move in the middlegame range
                    move_num = random.randint(min_move, min(max_move, len(moves) - 1))
                    
                    # Play to that position
                    for i, move in enumerate(moves[:move_num]):
                        board.push(move)
                    
                    # Skip if game is over or too few legal moves
                    if not board.is_game_over() and len(list(board.legal_moves)) > 5:
                        positions.append(board.copy())
        
        except Exception as e:
            continue
    
    print(f"✓ Sampled {len(positions)} positions from pro games (seed={seed})")
    return positions


def _generate_benchmark_positions(
    num_positions: int, 
    source: str = "pro_games",
    seed: int = None
) -> List[chess.Board]:
    """Get benchmark positions from various sources.
    
    Args:
        num_positions: Number of positions to return
        source: Position source - "pro_games" (default), "frozen", or "random"
        seed: Random seed for reproducibility (only for pro_games)
    
    Returns:
        List of chess.Board objects
    """
    if source == "pro_games":
        # Use pro game positions (avoids overfitting to fixed positions)
        positions = sample_pro_game_positions(num_positions, seed=seed)
        if positions:
            return positions
        # Fallback to frozen if no PGN files found
        source = "frozen"
    
    if source == "frozen":
        # Use frozen positions file
        positions = load_frozen_positions()
        if positions:
            return positions[:num_positions]
    
    # Fallback: generate random positions
    positions = []
    for _ in range(num_positions * 2):
        board = chess.Board()
        num_moves = np.random.randint(10, 30)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(np.random.choice(legal_moves))
        
        if not board.is_game_over() and len(list(board.legal_moves)) > 5:
            positions.append(board.copy())
        
        if len(positions) >= num_positions:
            break
    
    return positions


# =============================================================================
# REGRESSION DETECTION
# =============================================================================

def detect_regression(
    current_score: float,
    baseline_score: float,
    threshold: float = 0.05
) -> Tuple[bool, str]:
    """
    Detect if model has regressed compared to baseline.
    
    Args:
        current_score: Current model's benchmark score
        baseline_score: Baseline model's benchmark score
        threshold: Regression threshold (fraction)
    
    Returns:
        (is_regressed, message)
    """
    if baseline_score == 0:
        return False, "No baseline available"
    
    delta = (current_score - baseline_score) / abs(baseline_score)
    
    if delta < -threshold:
        return True, f"⚠ REGRESSION: {delta*100:.1f}% decrease from baseline"
    elif delta > threshold:
        return False, f"✓ IMPROVEMENT: {delta*100:.1f}% increase from baseline"
    else:
        return False, f"○ STABLE: {delta*100:.1f}% change from baseline"


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Stockfish Evaluation Utilities")
    parser.add_argument("--benchmark", action="store_true", help="Run model benchmark")
    parser.add_argument("--positions", type=int, default=50, help="Number of positions")
    parser.add_argument("--model-path", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--stockfish-path", type=str, default=None, help="Stockfish executable")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish search depth")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("STOCKFISH EVALUATION UTILITIES")
    print("=" * 60)
    
    evaluator = StockfishEvaluator(
        stockfish_path=args.stockfish_path,
        depth=args.depth
    )
    
    if evaluator.engine is None:
        print("❌ Stockfish not found. Please specify --stockfish-path")
        return 1
    
    if args.benchmark:
        if args.model_path is None:
            print("❌ --model-path required for benchmarking")
            return 1
        
        # Load model
        from models import create_chess_model, load_model_with_compatibility
        import torch
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = create_chess_model("big").to(device)
        model = load_model_with_compatibility(model, args.model_path, device)
        
        print(f"\nBenchmarking on {args.positions} positions...")
        results = evaluate_model_vs_stockfish(
            model, device, evaluator,
            num_positions=args.positions,
            input_channels=22
        )
        
        print("\nResults:")
        print(f"  Value MAE: {results['value_mae']:.4f}")
        print(f"  Policy Agreement: {results['policy_agreement']*100:.1f}%")
        print(f"  Positions: {results['positions_evaluated']}")
    
    evaluator.close()
    return 0


if __name__ == "__main__":
    exit(main())
