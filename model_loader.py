"""
Unified Model Loader for Chess AI
==================================

This module provides a unified interface to load any trained model
from the train/ folder (limited, small, medium, big) or legacy models.

Usage:
    from model_loader import load_chess_model, ChessModelWrapper
    
    # Load specific model type
    model = load_chess_model("limited")  # or "small", "medium", "big"
    
    # Or auto-detect from checkpoint
    model = load_chess_model(checkpoint_path="train/checkpoints_limited/model_best.pt")
    
    # Get best move
    move = model.get_best_move(board, method="mcts")  # or "direct"
"""

import os
import sys
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# Add train folder to path for imports
TRAIN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train")
if TRAIN_PATH not in sys.path:
    sys.path.insert(0, TRAIN_PATH)

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# BOARD TO TENSOR CONVERSION (supports all model types)
# ============================================================================
def board_to_tensor(board, input_channels=18):
    """Convert chess board to tensor representation.
    
    Args:
        board: chess.Board object
        input_channels: 18 (limited/small), 20 (legacy), or 22 (medium/big)
    
    Returns:
        numpy array of shape (input_channels, 8, 8)
    """
    tensor = np.zeros((input_channels, 8, 8), dtype=np.float32)
    
    # Piece planes (channels 0-11)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                channel = piece_type - 1 if color == chess.WHITE else piece_type + 5
                tensor[channel, row, col] = 1
    
    # Castling rights (channels 12-15)
    tensor[12, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[13, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[15, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    
    # En passant (channel 16)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[16, row, col] = 1
    
    # Turn (channel 17)
    tensor[17, :, :] = 1 if board.turn == chess.WHITE else 0
    
    if input_channels >= 20:
        # Halfmove clock and move number (channels 18-19)
        tensor[18, :, :] = board.halfmove_clock / 50.0
        move_number = board.fullmove_number
        tensor[19, :, :] = min(move_number / 100.0, 1.0)
    
    if input_channels >= 22:
        # Attack maps (channels 20-21)
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            if board.is_attacked_by(chess.WHITE, square):
                tensor[20, row, col] = 1.0
            if board.is_attacked_by(chess.BLACK, square):
                tensor[21, row, col] = 1.0
    
    return tensor


# ============================================================================
# MOVE INDEX MAPPING
# ============================================================================
# Standard move encoding: from_square * 64 + to_square = indices 0-4095
# Promotion moves: indices 4096-4671

_promotion_moves = {}
_reverse_promotion_moves = {}
_promotion_idx = 4096

for rank in [6, 1]:  # Ranks where pawns can promote (7th for white, 2nd for black)
    for col in range(8):
        from_square = chess.square(col, rank)
        directions = [0]  # Straight ahead
        if col > 0:
            directions.append(-1)  # Capture left
        if col < 7:
            directions.append(1)   # Capture right
        
        for d in directions:
            to_col = col + d
            to_rank = rank + (1 if rank == 6 else -1)
            if 0 <= to_col < 8:
                to_square = chess.square(to_col, to_rank)
                for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                    _promotion_moves[(from_square, to_square, piece)] = _promotion_idx
                    _reverse_promotion_moves[_promotion_idx] = (from_square, to_square, piece)
                    _promotion_idx += 1


def get_move_index(move):
    """Convert a chess move to policy index."""
    if move.promotion:
        key = (move.from_square, move.to_square, move.promotion)
        return _promotion_moves.get(key, move.from_square * 64 + move.to_square)
    return move.from_square * 64 + move.to_square


def index_to_move(board, index):
    """Convert a policy index to a chess move."""
    legal_moves = list(board.legal_moves)
    
    if index < 4096:
        from_square = index // 64
        to_square = index % 64
        
        # Try regular move
        candidate = chess.Move(from_square, to_square)
        if candidate in legal_moves:
            return candidate
        
        # Try with promotion (queen first)
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            candidate = chess.Move(from_square, to_square, promotion=promo)
            if candidate in legal_moves:
                return candidate
    else:
        # Promotion move
        if index in _reverse_promotion_moves:
            from_sq, to_sq, promo = _reverse_promotion_moves[index]
            candidate = chess.Move(from_sq, to_sq, promotion=promo)
            if candidate in legal_moves:
                return candidate
    
    # Fallback to random legal move
    return random.choice(legal_moves) if legal_moves else None


# ============================================================================
# MCTS IMPLEMENTATION (Corrected PUCT formula)
# ============================================================================
class MCTSNode:
    """MCTS Node with correct PUCT formula."""
    __slots__ = ['board', 'parent', 'move', 'prior', 'children', 
                 'visits', 'total_value', 'is_expanded']
    
    def __init__(self, board, parent=None, move=None, prior=0.0):
        self.board = board.copy() if board else None
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.is_expanded = False
    
    def q_value(self):
        """Average value of this node."""
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits
    
    def ucb_score(self, c_puct=2.0):
        """Calculate UCB score using PUCT formula (AlphaZero style).
        
        PUCT = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(child))
        """
        if self.parent is None:
            return 0.0
        
        parent_visits = max(1, self.parent.visits)
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return self.q_value() + exploration
    
    def select_child(self, c_puct=2.0):
        """Select child with highest UCB score."""
        if not self.children:
            return None
        return max(self.children.values(), key=lambda c: c.ucb_score(c_puct))
    
    def expand(self, policy_probs, dirichlet_noise=None, noise_weight=0.25):
        """Expand node with all legal moves."""
        if self.is_expanded or self.board.is_game_over():
            return
        
        legal_moves = list(self.board.legal_moves)
        if not legal_moves:
            return
        
        # Add Dirichlet noise at root
        if dirichlet_noise is not None and len(dirichlet_noise) == len(legal_moves):
            noise = dirichlet_noise
        else:
            noise = [0] * len(legal_moves)
            noise_weight = 0
        
        for i, move in enumerate(legal_moves):
            idx = get_move_index(move)
            prior = policy_probs[idx] if idx < len(policy_probs) else 0.001
            
            # Mix with noise at root
            if noise_weight > 0:
                prior = (1 - noise_weight) * prior + noise_weight * noise[i]
            
            new_board = self.board.copy()
            new_board.push(move)
            self.children[move] = MCTSNode(new_board, self, move, prior)
        
        self.is_expanded = True
    
    def backpropagate(self, value):
        """Backpropagate value up the tree."""
        node = self
        while node is not None:
            node.visits += 1
            node.total_value += value
            value = -value  # Flip for opponent
            node = node.parent


# ============================================================================
# CHESS MODEL WRAPPER
# ============================================================================
class ChessModelWrapper:
    """Unified wrapper for any chess model."""
    
    def __init__(self, model, input_channels=18, model_type="limited"):
        self.model = model
        self.input_channels = input_channels
        self.model_type = model_type
        self.model.eval()
        self.mcts_tree = None  # For tree reuse
    
    def _get_policy_value(self, board):
        """Get policy and value from the model."""
        tensor = board_to_tensor(board, self.input_channels)
        input_tensor = torch.tensor(tensor).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, value = self.model(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        
        return policy_probs, value
    
    def get_best_move_direct(self, board, temperature=1.0):
        """Get best move using direct policy output."""
        policy_probs, _ = self._get_policy_value(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        move_scores = {}
        for move in legal_moves:
            idx = get_move_index(move)
            score = policy_probs[idx] if idx < len(policy_probs) else 0.001
            move_scores[move] = score
        
        if temperature < 0.01:
            return max(move_scores, key=move_scores.get)
        else:
            # Sample with temperature
            moves = list(move_scores.keys())
            scores = np.array([move_scores[m] for m in moves])
            scores = np.power(scores, 1.0 / temperature)
            probs = scores / scores.sum()
            return np.random.choice(moves, p=probs)
    
    def get_best_move_mcts(self, board, num_simulations=200, c_puct=2.0, 
                           temperature=0.1, dirichlet_alpha=0.3):
        """Get best move using MCTS with corrected PUCT formula."""
        
        # Create root node
        root = MCTSNode(board)
        
        # Get initial policy
        policy_probs, _ = self._get_policy_value(board)
        
        # Add Dirichlet noise at root for exploration
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
        root.expand(policy_probs, noise, noise_weight=0.25)
        
        # Run simulations
        for _ in range(num_simulations):
            node = root
            
            # Selection: traverse to leaf
            while node.is_expanded and node.children:
                node = node.select_child(c_puct)
                if node is None:
                    break
            
            if node is None:
                continue
            
            # Check terminal
            if node.board.is_game_over():
                if node.board.is_checkmate():
                    value = -1.0  # Loss for side to move
                else:
                    value = 0.0  # Draw
            else:
                # Expansion and evaluation
                policy_probs, value = self._get_policy_value(node.board)
                node.expand(policy_probs)
                value = -value  # From opponent's perspective
            
            # Backpropagation
            node.backpropagate(value)
        
        # Select move based on visit counts
        if not root.children:
            return random.choice(legal_moves)
        
        visit_counts = np.array([child.visits for child in root.children.values()])
        moves = list(root.children.keys())
        
        if temperature < 0.01:
            return moves[np.argmax(visit_counts)]
        else:
            visit_counts = np.power(visit_counts.astype(float), 1.0 / temperature)
            probs = visit_counts / visit_counts.sum()
            return np.random.choice(moves, p=probs)
    
    def get_best_move(self, board, method="mcts", **kwargs):
        """Get best move using specified method."""
        if method == "direct":
            return self.get_best_move_direct(board, **kwargs)
        else:
            return self.get_best_move_mcts(board, **kwargs)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
def load_chess_model(model_type="limited", checkpoint_path=None, device_override=None):
    """Load a chess model by type or from checkpoint.
    
    Args:
        model_type: "limited", "small", "medium", or "big"
        checkpoint_path: Optional path to specific checkpoint
        device_override: Optional device override
    
    Returns:
        ChessModelWrapper instance ready for inference
    """
    global device
    if device_override:
        device = torch.device(device_override)
    
    # Try to import from train folder
    try:
        from models import create_chess_model, LimitedChessNet, ChessNet
    except ImportError:
        # Fallback: define models inline
        print("Warning: Could not import from train/models.py, using inline definitions")
        return _load_legacy_model(checkpoint_path)
    
    # Determine input channels based on model type
    channel_map = {
        "limited": 18,
        "small": 18,
        "medium": 22,
        "big": 22,
    }
    input_channels = channel_map.get(model_type.lower(), 18)
    
    # Create model
    model = create_chess_model(model_type)
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        # Try to find default checkpoint
        default_paths = [
            f"train/checkpoints_{model_type}/model_best.pt",
            f"train/checkpoints_{model_type}/model_epoch_0100.pt",
            f"train/chess_model/chess_model_{model_type}.pth",
            "chess_model/chess_model.pth",
        ]
        
        for path in default_paths:
            full_path = os.path.join(os.path.dirname(__file__), path)
            if os.path.exists(full_path):
                print(f"Loading model from {full_path}")
                checkpoint = torch.load(full_path, map_location=device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                break
        else:
            print(f"Warning: No checkpoint found for {model_type} model. Using random weights.")
    
    model.eval()
    return ChessModelWrapper(model, input_channels, model_type)


def _load_legacy_model(checkpoint_path):
    """Load legacy model when train/ imports fail."""
    from pytorch_model import ChessNet, PytorchModel
    
    if checkpoint_path:
        return PytorchModel(checkpoint_path)
    return PytorchModel()


def list_available_models():
    """List all available model checkpoints."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models = []
    
    # Check train folder checkpoints
    for model_type in ["limited", "small", "medium", "big"]:
        checkpoint_dir = os.path.join(base_dir, "train", f"checkpoints_{model_type}")
        if os.path.exists(checkpoint_dir):
            best = os.path.join(checkpoint_dir, "model_best.pt")
            if os.path.exists(best):
                models.append({
                    "type": model_type,
                    "path": best,
                    "name": f"{model_type} (best)"
                })
    
    # Check legacy models
    legacy_paths = [
        ("chess_model/chess_model.pth", "Legacy Default"),
        ("chess_model/old/100000games/chess_model.pth", "Legacy 100k Games"),
    ]
    
    for path, name in legacy_paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            models.append({
                "type": "legacy",
                "path": full_path,
                "name": name
            })
    
    return models


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    print("Available models:")
    for m in list_available_models():
        print(f"  - {m['name']}: {m['path']}")
    
    print("\nTesting model loading...")
    try:
        model = load_chess_model("limited")
        board = chess.Board()
        move = model.get_best_move(board, method="direct")
        print(f"Direct policy move: {move}")
        
        move = model.get_best_move(board, method="mcts", num_simulations=50)
        print(f"MCTS move (50 sims): {move}")
    except Exception as e:
        print(f"Error: {e}")
