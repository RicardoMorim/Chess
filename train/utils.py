import gc
import torch
import torch.nn.functional as F
import chess
import numpy as np
from typing import List, Tuple

from constants import TACTICAL_TEST_POSITIONS
from data import board_to_tensor, get_move_index


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================
def clear_memory():
    """Force garbage collection and clear CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_optimal_batch_size(model, device, starting_size=64, min_size=8, max_size=256):
    """Find the largest batch size that fits in memory.
    
    Uses binary search to efficiently find the optimal batch size.
    
    Args:
        model: The neural network model
        device: Computation device
        starting_size: Initial batch size to try
        min_size: Minimum acceptable batch size
        max_size: Maximum batch size to consider
    
    Returns:
        Optimal batch size that fits in memory
    """
    input_channels = getattr(model, 'input_channels', 22)
    
    # Binary search for optimal size
    low, high = min_size, min(starting_size * 2, max_size)
    best_size = min_size
    
    model.eval()
    
    while low <= high:
        mid = (low + high) // 2
        try:
            dummy_input = torch.randn(mid, input_channels, 8, 8, device=device)
            with torch.no_grad():
                model(dummy_input)
            del dummy_input
            clear_memory()
            
            # Success - try larger
            best_size = mid
            low = mid + 1
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                high = mid - 1
                clear_memory()
            else:
                raise e
    
    return best_size


# ============================================================================
# TACTICAL TESTING
# ============================================================================

# Cache for loaded puzzles to avoid reloading every test
_cached_puzzles = None

def load_tactical_test_positions(num_samples=50, cache_dir="./cache") -> List[Tuple[str, str, str]]:
    """Load random tactical test positions from Lichess puzzle cache.
    
    Samples puzzles from the same distribution as training for accurate evaluation.
    Prioritizes mate puzzles but includes a mix of other tactics.
    
    Args:
        num_samples: Number of test positions to sample
        cache_dir: Directory containing puzzle caches
    
    Returns:
        List of (fen, best_move_uci, category) tuples
    """
    global _cached_puzzles
    import os
    import pickle
    import random
    
    # Try to load from Lichess cache first
    if _cached_puzzles is None:
        # Look for Lichess puzzle cache files
        cache_files = []
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) 
                          if f.startswith('lichess_puzzles_') and f.endswith('.pkl')]
        
        if cache_files:
            # Use most recent cache file
            cache_file = os.path.join(cache_dir, sorted(cache_files)[-1])
            try:
                with open(cache_file, 'rb') as f:
                    _cached_puzzles = pickle.load(f)
                print(f"Loaded {len(_cached_puzzles)} puzzles for testing from cache")
            except Exception as e:
                print(f"Error loading puzzle cache: {e}")
                _cached_puzzles = []
        else:
            print("No Lichess cache found, using fallback hardcoded positions")
            # Fallback to hardcoded positions if no cache
            from constants import TACTICAL_TEST_POSITIONS
            all_positions = []
            for category, positions in TACTICAL_TEST_POSITIONS.items():
                for fen, best_move in positions:
                    all_positions.append((fen, best_move, category))
            return all_positions
    
    if not _cached_puzzles:
        return []
    
    # Sample puzzles with category distribution matching training priorities
    # Mate puzzles should be well represented
    test_positions = []
    
    # Group puzzles by category
    by_category = {}
    for puzzle in _cached_puzzles:
        if len(puzzle) >= 4:
            fen, move, value, category = puzzle[:4]
        else:
            fen, move, value = puzzle[:3]
            category = "other"
        
        if category not in by_category:
            by_category[category] = []
        by_category[category].append((fen, move, category))
    
    # Sample distribution: prioritize mates but include tactics
    sample_weights = {
        'mate_in_one': 0.25,
        'mate_in_two': 0.15,
        'mate_in_three': 0.10,
        'mate_longer': 0.05,
        'fork': 0.10,
        'pin': 0.08,
        'discovered': 0.07,
        'skewer': 0.05,
        'endgame': 0.10,
        'other': 0.05
    }
    
    for category, weight in sample_weights.items():
        if category in by_category and by_category[category]:
            n_samples = max(1, int(num_samples * weight))
            samples = random.sample(
                by_category[category], 
                min(n_samples, len(by_category[category]))
            )
            test_positions.extend(samples)
    
    # Shuffle and limit
    random.shuffle(test_positions)
    test_positions = test_positions[:num_samples]
    
    return test_positions


def test_tactical_recognition(model, device, verbose=True):
    """Test if model can recognize basic tactical patterns.
    
    This is a key metric for chess AI quality - the ability to find
    tactical shots like forks, pins, and checkmates.
    
    Args:
        model: The neural network
        device: Computation device
        verbose: Whether to print detailed results
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    model.eval()
    
    test_positions = load_tactical_test_positions()
    batch_size = 8
    correct = 0
    correct_by_category = {}
    total_by_category = {}
    
    input_channels = getattr(model, 'input_channels', 22)
    
    for i in range(0, len(test_positions), batch_size):
        batch_positions = test_positions[i:i+batch_size]
        boards = [chess.Board(fen) for fen, _, _ in batch_positions]
        best_moves = [move_uci for _, move_uci, _ in batch_positions]
        categories = [cat for _, _, cat in batch_positions]
        
        # Batch process
        input_tensors = torch.stack([
            torch.tensor(board_to_tensor(board, 0, input_channels), dtype=torch.float32)
            for board in boards
        ]).to(device)
        
        with torch.no_grad():
            policy_logits, _ = model(input_tensors)
        
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        
        for j, (board, best_move_uci, policy, category) in enumerate(
            zip(boards, best_moves, policies[:len(boards)], categories)
        ):
            # Track by category
            if category not in correct_by_category:
                correct_by_category[category] = 0
                total_by_category[category] = 0
            total_by_category[category] += 1
            
            legal_moves = list(board.legal_moves)
            move_probs = np.zeros(len(legal_moves))
            best_move_idx = -1
            
            for idx, move in enumerate(legal_moves):
                move_idx = get_move_index(move)
                if move_idx < len(policy):
                    move_probs[idx] = policy[move_idx]
                if move.uci() == best_move_uci:
                    best_move_idx = idx
            
            if legal_moves:
                top_move_idx = np.argmax(move_probs)
                if top_move_idx == best_move_idx:
                    correct += 1
                    correct_by_category[category] += 1
                    if verbose:
                        print(f"✓ [{category}] Correct: {best_move_uci}")
                else:
                    if verbose:
                        pred_move = legal_moves[top_move_idx].uci()
                        print(f"✗ [{category}] Expected: {best_move_uci}, Got: {pred_move}")
    
    # Print summary
    total = len(test_positions)
    print(f"\n{'='*40}")
    print(f"Tactical Recognition: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"{'='*40}")
    for category in sorted(total_by_category.keys()):
        cat_correct = correct_by_category[category]
        cat_total = total_by_category[category]
        print(f"  {category}: {cat_correct}/{cat_total} ({100*cat_correct/cat_total:.1f}%)")
    
    return correct / total if total > 0 else 0.0


# ============================================================================
# MODEL UTILITIES
# ============================================================================
def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model):
    """Print a summary of the model architecture."""
    print(f"\n{'='*50}")
    print("MODEL SUMMARY")
    print(f"{'='*50}")
    print(f"Architecture: ChessNet")
    print(f"Input channels: {getattr(model, 'input_channels', 'N/A')}")
    print(f"Residual blocks: {getattr(model, 'num_blocks', 'N/A')}")
    print(f"Hidden channels: {getattr(model, 'channels', 256)}")
    print(f"SE blocks: {getattr(model, 'use_se', False)}")
    print(f"Total parameters: {count_parameters(model):,}")
    print(f"{'='*50}\n")
