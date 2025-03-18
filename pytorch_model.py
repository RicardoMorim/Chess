import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import glob
import json
import signal
import sys
import itertools
import math
import random

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Residual Block for the neural network
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

# Chess Neural Network with updated architecture
class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(20, channels, kernel_size=3, padding=1)  # 20 channels for added features
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 73 * 8 * 8)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

# Convert chess board to input tensor with enhanced features
def board_to_tensor(board, move_number=None):
    if move_number is None:
        # Estimate move number from board state if not provided
        move_number = (board.fullmove_number * 2) - (2 if board.turn == chess.WHITE else 1)
    
    tensor = np.zeros((20, 8, 8), dtype=np.float32)  # Increased to 20 channels
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                channel = piece_type - 1 if color == chess.WHITE else piece_type + 5
                tensor[channel, row, col] = 1
    tensor[12, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[13, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[15, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[16, row, col] = 1
    tensor[17, :, :] = 1 if board.turn == chess.WHITE else 0
    tensor[18, :, :] = board.halfmove_clock / 50.0  # Normalized repetition counter 
    tensor[19, :, :] = move_number / 200.0  # Normalized move number (assuming max 200 moves)
    return tensor

# Move Index Mapping with Promotions
promotion_moves = {}
promotion_idx = 4096
for rank in [6, 1]:
    for col in range(8):
        from_square = chess.square(col, rank)
        to_square = chess.square(col, rank + (1 if rank == 6 else -1))
        for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promotion_moves[(from_square, to_square, piece)] = promotion_idx
            promotion_idx += 1
        if col > 0:
            to_square = chess.square(col - 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1
        if col < 7:
            to_square = chess.square(col + 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1

def get_move_index(move):
    if move.promotion:
        return promotion_moves[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square

def index_to_move(board, index):
    """
    Convert a policy index (0–4671) into a chess.Move object for the given board.
    """
    legal_moves = list(board.legal_moves)

    # Precompute reverse mapping for promotion moves once
    if not hasattr(index_to_move, "reverse_promotion_moves"):
        # Create a mapping: index -> (from_square, to_square, promotion)
        index_to_move.reverse_promotion_moves = {v: k for k, v in promotion_moves.items()}
    reverse_promotion_moves = index_to_move.reverse_promotion_moves

    # For non-promotion moves: indices 0–4095
    if index < 4096:
        from_square = index // 64
        to_square = index % 64
        candidate = chess.Move(from_square, to_square)
        if candidate in legal_moves:
            return candidate

        # Check if a promotion move might be legal from that square
        for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            candidate = chess.Move(from_square, to_square, promotion=promo_piece)
            if candidate in legal_moves:
                return candidate

    # For promotion moves: indices 4096–4671
    else:
        if index in reverse_promotion_moves:
            from_square, to_square, promo_piece = reverse_promotion_moves[index]
            candidate = chess.Move(from_square, to_square, promotion=promo_piece)
            if candidate in legal_moves:
                return candidate

        # Fallback: if reverse mapping doesn't yield a legal move, pick one from legal promotion moves.
        promotion_moves_in_board = [move for move in legal_moves if move.promotion is not None]
        if promotion_moves_in_board:
            return promotion_moves_in_board[0]

    # As a final fallback, return a random legal move.
    return random.choice(legal_moves)

# MCTS Node
class MCTSNode:
    def __init__(self, board, parent=None, move=None, prior=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.untried_moves = list(board.legal_moves)

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def ucb1(self, c=1.4):
        if self.visits == 0:
            return float('inf')
        # Avoid unnecessary computation
        exploitation = self.total_value / self.visits
        parent_visits = max(1, self.parent.visits if self.parent else 1)
        exploration = c * self.prior * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def select_child(self, c=1.4):
        if not self.children:
            return None
        return max(self.children.items(), key=lambda item: item[1].ucb1(c=c))[1]

    def expand(self):
        if not self.untried_moves:
            return None
        move = self.untried_moves.pop() 
        new_board = self.board.copy()
        new_board.push(move)
        child = MCTSNode(new_board, self, move)
        self.children[move] = child
        return child

    def simulate(self):
        input_tensor = torch.tensor(board_to_tensor(self.board)).unsqueeze(0).to(device)
        with torch.no_grad():
            _, value = self.model(input_tensor)
        return value.item()

    def backpropagate(self, value):
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(-value)


# Direct Move Selection
def direct_select_move(board, model, temperature=1.2):
    model.eval()
    with torch.no_grad():
        # Get the move number for the enhanced features
        move_number = (board.fullmove_number * 2) - (2 if board.turn == chess.WHITE else 1)
        input_tensor = torch.tensor(board_to_tensor(board, move_number)).unsqueeze(0).to(device)
        policy_logits, _ = model(input_tensor)
        policy_probs = F.softmax(policy_logits / temperature, dim=1).cpu().numpy()[0]
        legal_moves = list(board.legal_moves)
        move_scores = {}
        for move in legal_moves:
            idx = get_move_index(move)
            score = policy_probs[idx] if idx < len(policy_probs) else 0
            move_scores[move] = score
            if move.promotion:
                print(f"Promotion move: {move.uci()}, Index: {idx}, Score: {score}")
        if not move_scores:
            return random.choice(legal_moves)
        best_move = max(move_scores, key=move_scores.get)
        print(f"Selected move: {best_move.uci()}, Score: {move_scores[best_move]}")
        return best_move

class PytorchModel:
    def __init__(self, model_path="./chess_model/chess_model.pth"):
        # Updated to use the enhanced model with 10 blocks
        self.model = ChessNet(num_blocks=10, channels=256).to(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"Model loaded from {model_path}")

    def best_move_direct(self, board, temperature=1.2):
        piece_count = sum(len(board.pieces(piece_type, color)) 
                          for piece_type in chess.PIECE_TYPES 
                          for color in chess.COLORS)
        if piece_count <= 6 or board.can_claim_draw():
            return self.get_best_move_mtcs(board, iterations=5000, c=2.5, dirichlet_alpha=0.1)
        return direct_select_move(board, self.model, temperature=temperature)

    def get_best_move_mtcs(self, board, iterations=10000, c=2.0, dirichlet_alpha=0.03):
        root = MCTSNode(board)
        self.model.eval()

        # Get the move number for enhanced features
        move_number = (board.fullmove_number * 2) - (2 if board.turn == chess.WHITE else 1)
        input_tensor = torch.tensor(board_to_tensor(board, move_number)).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _ = self.model(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

        legal_moves = list(board.legal_moves)
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            idx = get_move_index(move)
            prior = policy_probs[idx] if idx < len(policy_probs) else 0.001
            prior = 0.75 * prior + 0.25 * noise[i]
            root.children[move] = MCTSNode(board.copy(), root, move, prior=prior)
        root.untried_moves = []

        batch_size = 8
        batch = []
        for i in range(iterations):
            if i % 500 == 0 and i > 2000:
                best_move = max(root.children, key=lambda m: root.children[m].visits)
                best_visits = root.children[best_move].visits
                total_visits = sum(child.visits for child in root.children.values())
                if best_visits > total_visits * 0.9:
                    print(f"Early stopping after {i} iterations")
                    return best_move

            node = root
            while node.is_fully_expanded() and node.children:
                node = node.select_child(c=c)
            if node and not node.is_fully_expanded():
                child = node.expand()
                if child:
                    batch.append(child)
            if len(batch) >= batch_size or i == iterations - 1 and batch:
                # Add the move number for each board in the batch
                batch_arrays = np.array([board_to_tensor(
                    node.board, 
                    (node.board.fullmove_number * 2) - (2 if node.board.turn == chess.WHITE else 1)
                ) for node in batch])
                inputs = torch.from_numpy(batch_arrays).to(device)
                with torch.no_grad():
                    _, values = self.model(inputs)
                for node, value in zip(batch, values):
                    node.backpropagate(value.item())
                batch = []

        return max(root.children, key=lambda m: root.children[m].visits)