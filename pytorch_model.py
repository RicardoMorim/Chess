import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import math
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import time

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

    def quantize(self):
        """Quantize the model for faster inference"""
        if not hasattr(torch, 'quantization'):
            print("PyTorch quantization not available")
            return self
        
        # Specify which layers to not quantize (BatchNorm)
        quantize_config = torch.quantization.get_default_qconfig('fbgemm')
        self.qconfig = quantize_config
        
        # Prepare the model for quantization
        torch.quantization.prepare(self, inplace=True)
        
        # Convert to quantized model
        torch.quantization.convert(self, inplace=True)
        print("Model quantized for faster inference")
        return self

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
        self.board = board.copy() if board else None
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.virtual_loss = 0  # For parallel MCTS
        self.untried_moves = list(board.legal_moves) if board else []
        self.lock = threading.Lock()  # Thread safety

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def ucb1(self, c=1.4):
        with self.lock:
            if self.visits == 0:
                return float('inf')
            exploitation = self.total_value / (self.visits + self.virtual_loss)
            parent_visits = max(1, self.parent.visits if self.parent else 1)
            exploration = c * self.prior * math.sqrt(math.log(parent_visits) / (self.visits + self.virtual_loss))
            return exploitation + exploration

    def select_child(self, c=1.4):
        if not self.children:
            return None
        
        best_score = -float('inf')
        best_child = None
        best_move = None
        
        for move, child in self.children.items():
            score = child.ucb1(c=c)
            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
                
        return best_child

    def expand(self):
        with self.lock:
            if not self.untried_moves:
                return None
            move = self.untried_moves.pop()
            
        new_board = self.board.copy()
        new_board.push(move)
        
        # Child gets added with lock protection
        with self.lock:
            child = MCTSNode(new_board, self, move)
            self.children[move] = child
        return child

    def backpropagate(self, value):
        node = self
        while node:
            with node.lock:
                node.visits += 1
                node.total_value += value
                # Remove virtual loss that was applied during selection
                node.virtual_loss = max(0, node.virtual_loss - 1)
            value = -value  # Negate for opponent's perspective
            node = node.parent

    def apply_virtual_loss(self, amount=1):
        node = self
        while node:
            with node.lock:
                node.virtual_loss += amount
            node = node.parent



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

        self.model.quantize()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
        self.mcts_tree = None  # For tree reuse between moves

    def best_move_direct(self, board, temperature=1.2):
        piece_count = sum(len(board.pieces(piece_type, color)) 
                          for piece_type in chess.PIECE_TYPES 
                          for color in chess.COLORS)
        if piece_count <= 6 or board.can_claim_draw():
            return self.get_best_move_mcts(board, iterations=5000, c_puct=2.5, dirichlet_alpha=0.1)
        return direct_select_move(board, self.model, temperature=temperature)

    def get_best_move_mcts(self, board, iterations=10000, c_puct=2.0, dirichlet_alpha=0.03, 
                           temperature=1.0, parallel_workers=4, reuse_tree=True):
        """
        Get the best move using Monte Carlo Tree Search with parallelization and tree reuse.
        
        Args:
            board: The chess board position
            iterations: Number of MCTS simulations to run
            c_puct: Exploration constant for UCB formula
            dirichlet_alpha: Parameter for Dirichlet noise at root
            temperature: Temperature for move selection
            parallel_workers: Number of parallel workers for MCTS
            reuse_tree: Whether to reuse the tree from previous searches
        
        Returns:
            The best move from the current position
        """
        # Check if we can reuse the tree
        if reuse_tree and self.mcts_tree is not None:
            # If the last opponent move is in our tree, we can reuse it
            last_move = board.peek() if board.move_stack else None
            if last_move and last_move in self.mcts_tree.children:
                root = self.mcts_tree.children[last_move]
                root.parent = None  # Detach from parent
                print("Reusing subtree from previous search")
            else:
                # Create a new tree
                root = MCTSNode(board)
                print("Creating new search tree")
        else:
            # Create a new tree
            root = MCTSNode(board)
            
        self.model.eval()

        # Get the move number for enhanced features
        move_number = (board.fullmove_number * 2) - (2 if board.turn == chess.WHITE else 1)
        input_tensor = torch.tensor(board_to_tensor(board, move_number)).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _ = self.model(input_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]

        # Initialize children with priors and add Dirichlet noise at root
        legal_moves = list(board.legal_moves)
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            idx = get_move_index(move)
            prior = policy_probs[idx] if idx < len(policy_probs) else 0.001
            # Mix prior with noise (like AlphaZero)
            prior = 0.75 * prior + 0.25 * noise[i]
            
            new_board = board.copy()
            new_board.push(move)
            root.children[move] = MCTSNode(new_board, root, move, prior=prior)
            
        # Clear untried moves since we've already created all children
        root.untried_moves = []
        
        # Use a thread pool for parallel simulations
        def run_single_simulation():
            self._simulate_mcts(root, c_puct)
            
        # Run simulations in parallel
        batch_size = parallel_workers * 2  # Process more simulations than workers
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            for _ in range(0, iterations, batch_size):
                # Submit batch_size tasks
                futures = [executor.submit(run_single_simulation) for _ in range(min(batch_size, iterations - _))]
                # Wait for all current simulations to complete
                for future in futures:
                    future.result()
                    
                # Early stopping check (every batch)
                if _ >= 2000:  # Only check after sufficient iterations
                    best_move = max(root.children.items(), key=lambda x: x[1].visits)[0]
                    best_visits = root.children[best_move].visits
                    total_visits = sum(child.visits for child in root.children.values())
                    if best_visits > total_visits * 0.9:  # If one move has >90% of visits
                        print(f"Early stopping after {_} iterations")
                        break

        # Select move based on visit counts and temperature
        visit_counts = np.array([child.visits for child in root.children.values()])
        moves = list(root.children.keys())
        
        if temperature < 0.01:  # As temperature approaches 0, become deterministic
            best_idx = np.argmax(visit_counts)
            chosen_move = moves[best_idx]
        else:
            # Apply temperature and sample
            visit_counts = np.power(visit_counts, 1.0 / temperature)
            probs = visit_counts / np.sum(visit_counts)
            chosen_idx = np.random.choice(len(moves), p=probs)
            chosen_move = moves[chosen_idx]
            
        # Print move statistics
        total_visits = sum(child.visits for child in root.children.values())
        print(f"Move selection (T={temperature:.2f}):")
        for move, child in sorted(root.children.items(), key=lambda x: x[1].visits, reverse=True)[:5]:
            visit_pct = child.visits / total_visits * 100
            win_rate = (child.total_value / child.visits + 1) / 2 * 100 if child.visits > 0 else 0
            print(f"{move.uci()}: {visit_pct:.1f}% visits, {win_rate:.1f}% win rate")
            
        # Save tree for reuse (the chosen move becomes the new root)
        self.mcts_tree = root.children[chosen_move]
        self.mcts_tree.parent = None  # Detach from parent
        
        return chosen_move

    def _simulate_mcts(self, node, c_puct):
        """
        Run a single MCTS simulation from the given node.
        
        Args:
            node: The MCTSNode to start simulation from
            c_puct: Exploration constant
        
        Returns:
            The value estimate for the current position
        """
        # Terminal position check
        if node.board.is_game_over():
            if node.board.is_checkmate():
                # Return -1 when checkmate (because the side to move has lost)
                return -1.0
            else:
                # Draw or stalemate
                return 0.0
                
        # Selection: Traverse tree until we reach a leaf or unexpanded node
        if node.is_fully_expanded() and node.children:
            # Apply virtual loss to discourage other threads from taking same path
            node.apply_virtual_loss()
            child = node.select_child(c=c_puct)
            value = -self._simulate_mcts(child, c_puct)
            return value
            
        # Expansion: If the node is not fully expanded, expand it
        if not node.is_fully_expanded():
            child = node.expand()
            # If expansion returns None, this could be a terminal node we didn't catch earlier
            if not child:
                return 0.0
                
            # Simulation: Evaluate the new position
            move_number = (child.board.fullmove_number * 2) - (2 if child.board.turn == chess.WHITE else 1)
            input_tensor = torch.tensor(board_to_tensor(child.board, move_number)).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, value = self.model(input_tensor)
                
            child_value = -float(value.item())  # Negate because it's from opponent's perspective
            
            # Backpropagation
            child.backpropagate(child_value)
            return child_value
            
        # If we somehow get here, return a neutral value
        return 0.0