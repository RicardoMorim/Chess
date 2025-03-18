import gc
import torch
import torch.nn.functional as F
import chess
import numpy as np
from typing import List, Tuple

from constants import TACTICAL_TEST_POSITIONS
from data import board_to_tensor, get_move_index


def clear_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_optimal_batch_size(model, device, starting_size=32, min_size=8):
    """Find the largest batch size that fits in memory"""
    batch_size = starting_size
    
    while batch_size >= min_size:
        try:
            # Try to create a batch of random data
            dummy_input = torch.randn(batch_size, 20, 8, 8, device=device)
            model(dummy_input)
            dummy_input = None
            clear_memory()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                clear_memory()
            else:
                raise e
    
    return min_size  # Fallback to minimum size


def load_tactical_test_positions() -> List[Tuple[str, str, str]]:
    """Returns a flattened list of all tactical test positions"""
    all_positions = []
    for category, positions in TACTICAL_TEST_POSITIONS.items():
        for fen, best_move in positions:
            all_positions.append((fen, best_move, category))
    return all_positions


def test_tactical_recognition(model, device):
    """Test if model can recognize basic tactical patterns with batch processing"""
    model.eval()
    
    test_positions = load_tactical_test_positions()
    batch_size = 8  # Process multiple positions at once
    correct = 0
    
    for i in range(0, len(test_positions), batch_size):
        batch_positions = test_positions[i:i+batch_size]
        boards = [chess.Board(fen) for fen, _, _ in batch_positions]
        best_moves = [move_uci for _, move_uci, _ in batch_positions]
        
        # Batch process the input tensors
        input_tensors = torch.stack([
            torch.tensor(board_to_tensor(board, 0), dtype=torch.float32)
            for board in boards
        ]).to(device)
        
        with torch.no_grad():
            policy_logits, _ = model(input_tensors)
        
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        
        for j, (board, best_move_uci, policy) in enumerate(zip(boards, best_moves, policies[:len(boards)])):
            legal_moves = list(board.legal_moves)
            move_probs = np.zeros(len(legal_moves))
            best_move_idx = -1
            
            for idx, move in enumerate(legal_moves):
                move_idx = get_move_index(move)
                move_probs[idx] = policy[move_idx]
                if move.uci() == best_move_uci:
                    best_move_idx = idx
            
            if legal_moves:
                top_move_idx = np.argmax(move_probs)
                if top_move_idx == best_move_idx:
                    correct += 1
                    print(f"✓ Correct: {best_move_uci}")
                else:
                    print(f"✗ Expected: {best_move_uci}, Got: {legal_moves[top_move_idx].uci()}")
    
    print(f"Tactical test results: {correct}/{len(test_positions)} correct")
    return correct / len(test_positions)
