import time
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
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import gc
import pickle
import hashlib



def get_optimal_batch_size(starting_size=32, min_size=8):
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

# Dictionary of tactical test positions with categories
TACTICAL_TEST_POSITIONS = {
    # Checkmate patterns
    "mate_in_one": [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "h5f7"),
        ("r1bq2r1/ppp1bpkp/2np1np1/4p3/2B1P3/2NP1N2/PPPBQPPP/R3K2R w KQ - 0 1", "d2h6"),
        ("r3k2r/ppp2p1p/2n1bN2/2b1P1p1/2p1q3/2P5/PP1Q1PPP/RNB1K2R w KQkq - 0 1", "d2d8"),
    ],
    
    # Knight forks
    "knight_fork": [
        ("r3k2r/ppp2ppp/2n5/3Nn3/8/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5f6"),
        ("rnbqk2r/ppp1bppp/3p1n2/4p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1", "f3e5"),
        ("r1bqkbnr/ppp2ppp/2np4/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "f3e5"),
    ],
    
    # Pin patterns
    "pin": [
        ("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "c4f7"),
        ("rnbqkb1r/pp3ppp/2p1pn2/3p4/3P4/2NBPN2/PPP2PPP/R1BQK2R w KQkq - 0 1", "c3e5"),
        ("r1bqk2r/ppp2ppp/2n2n2/1B1pp3/1b2P3/2N2N2/PPPP1PPP/R1BQ1RK1 w kq - 0 1", "b5e8"),
    ],
    
    # Discovered attacks/checks
    "discovered": [
        ("rnbqkbnr/pppp1ppp/8/4p3/3P4/2N5/PPP1PPPP/R1BQKBNR b KQkq - 0 1", "e5d4"),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", "c4d5"),
        ("r1bqk2r/ppp1bppp/2n2n2/3pp3/2BP4/2N1PN2/PP3PPP/R1BQK2R w KQkq - 0 1", "d4e5"),
    ],
    
    # Skewers
    "skewer": [
        ("r1bqk1nr/ppp2ppp/2n5/3p4/1bBP4/2N5/PPP2PPP/R1BQK1NR w KQkq - 0 1", "c1g5"),
        ("r3k2r/pp3ppp/2p1bn2/q2p4/3P4/2PBP3/PP1N1PPP/R2QK2R b KQkq - 0 1", "e6a2"),
        ("r1b1kb1r/pp3ppp/2nqpn2/3p4/3P4/2N1PN2/PP3PPP/R1BQK2R w KQkq - 0 1", "f1b5"),
    ],
    
    # Endgame tactics
    "endgame": [
        ("8/8/1KP5/3r4/8/8/8/k7 w - - 0 1", "c6c7"),
        ("8/4kp2/2p3p1/1p2P1P1/8/2P3K1/8/8 w - - 0 1", "g3f4"),
        ("8/5p2/5k2/p1p2P2/Pp6/1P4K1/8/8 w - - 0 1", "g3f4"),
    ]
}

def clear_memory():
    """Force garbage collection and clear CUDA cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()










def load_tactical_test_positions():
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
        
        for j, (board, best_move_uci, policy) in enumerate(zip(boards, best_moves, policies)):
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


def filter_and_prioritize_puzzles_cached(puzzles, cache_dir="./cache"):
    """Filter puzzles with caching to avoid repeated work"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a hash based on the puzzles to use as cache key
    puzzles_hash = hashlib.md5(str(len(puzzles)).encode()).hexdigest()[:10]
    cache_file = os.path.join(cache_dir, f"puzzle_cache_{puzzles_hash}.pkl")
    
    # If cache exists, load from it
    if os.path.exists(cache_file):
        print(f"Loading {len(puzzles)} prioritized puzzles from cache...")
        try:
            with open(cache_file, 'rb') as f:
                prioritized_puzzles = pickle.load(f)
            
            print(f"Loaded prioritized puzzles from cache: {len(prioritized_puzzles)} puzzles")
            # Return early if we successfully loaded from cache
            return prioritized_puzzles
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
    
    # If we get here, we need to prioritize puzzles
    print(f"Prioritizing {len(puzzles)} puzzles (this may take a while)...")
    
    # Continue with your existing prioritization code
    mate_puzzles = []
    fork_puzzles = []
    pin_puzzles = []
    other_puzzles = []
    
    # Process puzzles in batches to avoid memory issues
    batch_size = 10000
    for i in range(0, len(puzzles), batch_size):
        batch = puzzles[i:i+batch_size]
        
        for fen, move_uci, value_target in batch:
            # Try to identify puzzle type from FEN or other properties
            board = chess.Board(fen)
            
            # Check if this is a checkmate puzzle
            future_board = board.copy()
            try:
                move = chess.Move.from_uci(move_uci)
                future_board.push(move)
                
                if future_board.is_checkmate():
                    mate_puzzles.append((fen, move_uci, 1.0))  # Higher value for mate puzzles
                elif "fork" in fen.lower() or detect_fork(board, move):
                    fork_puzzles.append((fen, move_uci, 0.9))
                elif "pin" in fen.lower() or detect_pin(board, move):
                    pin_puzzles.append((fen, move_uci, 0.8))
                else:
                    other_puzzles.append((fen, move_uci, value_target))
            except Exception:
                # Skip invalid puzzles
                continue
        
        # Show progress
        if (i + batch_size) % 100000 == 0 or (i + batch_size) >= len(puzzles):
            print(f"Processed {min(i + batch_size, len(puzzles))}/{len(puzzles)} puzzles...")
    
    # Combine with priority - duplicate tactical puzzles to increase their frequency  
    prioritized_puzzles = (
        mate_puzzles * 5 +  # Repeat mate puzzles 5x
        fork_puzzles * 3 +  # Repeat fork puzzles 3x
        pin_puzzles * 3 +   # Repeat pin puzzles 3x
        other_puzzles
    )
    
    # Store only a reasonable subset for training if there are too many
    max_puzzles = 200000  # Set a reasonable limit
    if len(prioritized_puzzles) > max_puzzles:
        print(f"Too many puzzles ({len(prioritized_puzzles)}), randomly sampling {max_puzzles}")
        import random
        random.shuffle(prioritized_puzzles)
        prioritized_puzzles = prioritized_puzzles[:max_puzzles]
    
    print(f"Prioritized puzzles: {len(prioritized_puzzles)} (from {len(puzzles)} original puzzles)")
    print(f"  - Mate puzzles: {len(mate_puzzles)}")
    print(f"  - Fork puzzles: {len(fork_puzzles)}")
    print(f"  - Pin puzzles: {len(pin_puzzles)}")
    
    # Cache the results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(prioritized_puzzles, f)
        print(f"Cached prioritized puzzles to {cache_file}")
    except Exception as e:
        print(f"Error caching results: {e}")
    
    return prioritized_puzzles
# Simple detectors for fork and pin
def detect_fork(board, move):
    """Simple heuristic to detect if a move creates a fork"""
    # This is a simplified version - a real implementation would be more complex
    attacker_piece = board.piece_at(move.from_square)
    if not attacker_piece:
        return False
        
    # Knights are common forking pieces
    if attacker_piece.piece_type == chess.KNIGHT:
        future_board = board.copy()
        future_board.push(move)
        
        # Count how many pieces the knight attacks after the move
        attacked_pieces = 0
        for square in chess.SQUARES:
            piece = future_board.piece_at(square)
            if piece and piece.color != attacker_piece.color:
                if future_board.is_attacked_by(attacker_piece.color, square):
                    attacked_pieces += 1
        
        # If attacking 2+ pieces, likely a fork
        return attacked_pieces >= 2
    
    return False

def detect_pin(board, move):
    """Simple heuristic to detect if a move creates or exploits a pin"""
    # This is a simplified version - a real implementation would be more complex
    attacker_piece = board.piece_at(move.from_square)
    if not attacker_piece:
        return False
        
    # Bishops, rooks, and queens commonly create pins
    if attacker_piece.piece_type in [chess.BISHOP, chess.ROOK, chess.QUEEN]:
        future_board = board.copy()
        future_board.push(move)
        
        # Check for aligned pieces that might indicate a pin
        for direction in [1, -1, 8, -8, 7, -7, 9, -9]:  # All 8 directions
            target_square = move.to_square
            pieces_in_line = []
            
            # Look along the line
            while True:
                target_square += direction
                if target_square < 0 or target_square > 63:
                    break
                    
                # Check if we've moved off the logical board line/diagonal
                if (direction in [1, -1] and chess.square_file(target_square) != 
                    chess.square_file(target_square - direction)):
                    break
                
                piece = future_board.piece_at(target_square)
                if piece:
                    pieces_in_line.append((target_square, piece))
                    if len(pieces_in_line) >= 2:
                        # If we found two pieces and the second is a king, it might be a pin
                        if pieces_in_line[1][1].piece_type == chess.KING:
                            return True
                    break
        
    return False

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Residual Block
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

# Chess Neural Network with adjustable blocks and channels
class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256):  # Increased to 10 blocks
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
        policy = self.policy_bn(self.policy_conv(x))  
        policy = policy.view(-1, 73 * 8 * 8)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

# Updated board_to_tensor with repetition counter and move number
def board_to_tensor(board, move_number):
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
    tensor[18, :, :] = board.halfmove_clock / 50.0  # Normalized repetition counter (fifty-move rule)
    tensor[19, :, :] = move_number / 200.0  # Normalized move number (assuming max 200 moves)
    return tensor

# Move Index Mapping (unchanged)
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

# Chess Dataset with Symmetry Augmentation
class ChessDataset(Dataset):
    def __init__(self, games, augment=True):
        self.positions = []
        self.augment = augment
        
        for game in games:
            result_str = game.headers.get("Result", "*")
            if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                continue
                
            result = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result_str]
            board = game.board()
            move_number = 1
            
            # Store positions as compact data
            for move in game.mainline_moves():
                # Store compressed representation instead of full tensor
                fen = board.fen()
                policy_target = get_move_index(move)
                value_target = result if board.turn == chess.WHITE else -result
                self.positions.append((fen, move_number, policy_target, value_target))
                
                if self.augment:
                    mirrored_board = board.mirror()
                    mirrored_move = chess.Move(
                        chess.square_mirror(move.from_square),
                        chess.square_mirror(move.to_square),
                        move.promotion
                    )
                    mirrored_policy = get_move_index(mirrored_move)
                    self.positions.append((mirrored_board.fen(), move_number, mirrored_policy, value_target))
                    
                board.push(move)
                move_number += 1
                
    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, move_number, policy_target, value_target = self.positions[idx]
        board = chess.Board(fen)
        input_tensor = board_to_tensor(board, move_number)
        
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))
    

# Puzzle Dataset (unchanged for now)
class PuzzleDataset(Dataset):
    def __init__(self, puzzles):
        self.puzzles = puzzles

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        fen, move_uci, value_target = self.puzzles[idx]
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        input_tensor = board_to_tensor(board, 0)  # Move number not tracked in puzzles
        policy_target = get_move_index(move)
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

# Load puzzles (unchanged)
def load_puzzles(pgn_file):
    puzzles = []
    with open(pgn_file, encoding='ISO-8859-1') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            try:
                best_move = list(game.mainline_moves())[0]
                fen = board.fen()
                move_uci = best_move.uci()
                puzzles.append((fen, move_uci, 1.0))
            except IndexError:
                continue
    return puzzles

def load_lichess_puzzles(csv_file):
    puzzles = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row['FEN']
            moves = row['Moves'].split()
            if moves:
                move_uci = moves[0]
                value_target = 1.0 if 'mate' in row['Themes'].lower() else 0.5
                puzzles.append((fen, move_uci, value_target))
    return puzzles

# Add this new loader function after your existing load_games_in_batches function
def load_professional_games(state_file, batch_size=1500, max_games=1500):
    """Load professional games more efficiently - one file at a time"""
    pro_pgn_directory = "./chess_pgns/pros"
    pro_pgn_files = glob.glob(os.path.join(pro_pgn_directory, "*.pgn"))
    
    if not pro_pgn_files:
        print("No professional games found in ./chess_pgns/pros/")
        return []
    
    # Track processed games and files
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_pro_games = state.get("processed_pro_games", 0)
            current_file_idx = state.get("current_pro_file_idx", 0)
            current_file_pos = state.get("current_pro_file_pos", 0)
        print(f"Resuming from {processed_pro_games} processed professional games")
    else:
        processed_pro_games = 0
        current_file_idx = 0
        current_file_pos = 0
    
    # Only process files that we need for this batch
    games = []
    
    while len(games) < batch_size and current_file_idx < len(pro_pgn_files):
        file = pro_pgn_files[current_file_idx]
        print(f"Loading professional games from {file}")
        
        with open(file) as pgn:
            # Seek to the previous position if continuing from last run
            if current_file_pos > 0:
                pgn.seek(current_file_pos)
            
            while len(games) < batch_size:
                # Save position before reading game
                pos = pgn.tell()
                game = chess.pgn.read_game(pgn)
                
                if game is None:
                    # End of file, move to next file
                    current_file_idx += 1
                    current_file_pos = 0
                    break
                
                # Save current position for next run
                current_file_pos = pgn.tell()
                games.append(game)
    
    # Save progress
    new_processed = processed_pro_games + len(games)
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
    else:
        state = {}
    
    state["processed_pro_games"] = new_processed
    state["current_pro_file_idx"] = current_file_idx
    state["current_pro_file_pos"] = current_file_pos
    
    with open(state_file, 'w') as f:
        json.dump(state, f)
    
    return games


# Improved load_games_in_batches function with position tracking
def load_games_in_batches(pgn_files, state_file, batch_size=1500):
    """Load regular games more efficiently - file position tracking"""
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_games = state.get("processed_games", 0)
            current_file_idx = state.get("current_file_idx", 0)
            current_file_pos = state.get("current_file_pos", 0)
        print(f"Resuming from {processed_games} processed games")
    else:
        processed_games = 0
        current_file_idx = 0
        current_file_pos = 0
    
    if current_file_idx >= len(pgn_files):
        print("All files processed. Starting over.")
        current_file_idx = 0
        current_file_pos = 0

    games = []
    
    while len(games) < batch_size and current_file_idx < len(pgn_files):
        file = pgn_files[current_file_idx]
        print(f"Loading games from {file}")
        
        with open(file) as pgn:
            # Seek to the previous position if continuing from last run
            if current_file_pos > 0:
                pgn.seek(current_file_pos)
            
            while len(games) < batch_size:
                # Save position before reading game
                pos = pgn.tell()
                game = chess.pgn.read_game(pgn)
                
                if game is None:
                    # End of file, move to next file
                    current_file_idx += 1
                    current_file_pos = 0
                    break
                
                # Save current position for next run
                current_file_pos = pgn.tell()
                games.append(game)
    
    # If we finished all files, wrap around
    if current_file_idx >= len(pgn_files) and len(games) < batch_size:
        current_file_idx = 0
        current_file_pos = 0
    
    new_processed = processed_games + len(games)
    
    state = {
        "processed_games": new_processed,
        "current_file_idx": current_file_idx,
        "current_file_pos": current_file_pos
    }
    
    with open(state_file, 'w') as f:
        json.dump(state, f)
    
    return games

# Self-play game generation (basic implementation)
def generate_self_play_games(model, num_games=100):
    games = []
    model.eval()
    
    # Pre-allocate tensor memory
    batch_size = min(16, num_games)  # Process up to 16 games in parallel
    active_games = [chess.Board() for _ in range(batch_size)]
    game_nodes = [chess.pgn.Game() for _ in range(batch_size)]
    current_nodes = [game for game in game_nodes]
    move_numbers = [1 for _ in range(batch_size)]
    active_mask = [True for _ in range(batch_size)]
    
    # Set headers
    for game in game_nodes:
        game.headers["Result"] = "*"
    
    while any(active_mask):
        # Get all active boards that aren't in terminal state
        active_indices = [i for i, active in enumerate(active_mask) if active]
        
        if not active_indices:
            break
            
        # Batch process all active boards
        input_tensors = torch.stack([
            torch.tensor(board_to_tensor(active_games[i], move_numbers[i]), dtype=torch.float32)
            for i in active_indices
        ]).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, _ = model(input_tensors)
        
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        
        # Process each active game
        for idx, i in enumerate(active_indices):
            board = active_games[i]
            legal_moves = list(board.legal_moves)
            
            if not legal_moves or board.is_game_over():
                # Game is over
                result = board.result()
                game_nodes[i].headers["Result"] = result
                games.append(game_nodes[i])
                
                # If we still need more games, start a new one
                if len(games) < num_games:
                    active_games[i] = chess.Board()
                    game_nodes[i] = chess.pgn.Game()
                    game_nodes[i].headers["Result"] = "*"
                    current_nodes[i] = game_nodes[i]
                    move_numbers[i] = 1
                else:
                    active_mask[i] = False
                continue
            
            # Get move probabilities
            policy = policies[idx]
            move_probs = np.zeros(len(legal_moves))
            
            for move_idx, move in enumerate(legal_moves):
                move_index = get_move_index(move)
                move_probs[move_idx] = policy[move_index]
            
            # Fix: Ensure we don't have all zeros by adding a small constant
            # and handle zero sums properly
            if np.sum(move_probs) <= 1e-10:
                # If all moves have essentially zero probability, use uniform distribution
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)
            else:
                # Normalize and handle potential division by zero
                move_probs = move_probs / np.sum(move_probs)
            
            # Select move
            move = np.random.choice(legal_moves, p=move_probs)
            
            # Apply move
            board.push(move)
            current_nodes[i] = current_nodes[i].add_variation(move)
            move_numbers[i] += 1
    
    # Add any remaining active games
    for i, active in enumerate(active_mask):
        if active:
            result = active_games[i].result()
            game_nodes[i].headers["Result"] = result
            games.append(game_nodes[i])
    
    return games[:num_games]  # Ensure we only return the requested number

# Update the train_batch function
def train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, epochs=5, processed_games=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Use the provided dataloaders directly
    puzzle_iter = itertools.cycle(puzzle_dataloader)
    
    # State loading
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            start_epoch = state.get("last_epoch", 0)
            print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        state = {"processed_games": processed_games, "last_epoch": 0}
        start_epoch = 0

    puzzle_frequency = 1
    puzzle_batch_multiplier = 10  
    policy_weight = 1.5
    value_weight = 1.0
    puzzle_policy_weight = 3.0  
    puzzle_value_weight = 2

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        game_batch_count = 0
        
        for game_batch in game_dataloader:
            inputs, policy_targets, value_targets = game_batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            # Use autocast for mixed precision
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, value_pred = model(inputs)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                loss = policy_weight * policy_loss + value_weight * value_loss
            
            # Scale gradients and optimize
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            game_batch_count += 1

            if game_batch_count % puzzle_frequency == 0:
                for _ in range(puzzle_batch_multiplier):
                    puzzle_batch = next(puzzle_iter)
                    inputs, policy_targets, value_targets = puzzle_batch
                    inputs = inputs.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    # Use autocast for puzzles too
                    with torch.amp.autocast(device_type="cuda"):
                        policy_logits, value_pred = model(inputs)
                        policy_loss = policy_loss_fn(policy_logits, policy_targets)
                        value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                        loss = puzzle_policy_weight * policy_loss + puzzle_value_weight * value_loss
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    total_loss += loss.item()

        # Rest of the epoch code (scheduler, saving) remains the same
        scheduler.step()
        num_puzzle_batches = len(game_dataloader) // puzzle_frequency * puzzle_batch_multiplier
        avg_loss = total_loss / (len(game_dataloader) + num_puzzle_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        state["last_epoch"] = epoch + 1
        state["processed_games"] = processed_games
        
        torch.save(model.state_dict(), save_path)
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"Checkpoint saved at epoch {epoch + 1}")


# Signal handler (updated)
def signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch):
    print("\nTraining interrupted! Saving model and state...")
    torch.save(model.state_dict(), save_path)
    with open(state_file, 'w') as f:
        state = {"processed_games": processed_games, "last_epoch": current_epoch + 1}
        json.dump(state, f)
    print(f"Model saved to {save_path}, state saved to {state_file}")
    sys.exit(0)

def train_tactical(model, optimizer, dataloader, device, epochs=3):
    """Train on tactical puzzles for a specific number of epochs"""
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        batch_count = 0
        total_loss = 0
        
        for batch in dataloader:
            inputs, policy_targets, value_targets = batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            policy_logits, value_pred = model(inputs)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            
            # Higher policy weight for tactical training
            loss = 3.0 * policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            if batch_count >= 10:  # Limit number of batches for speed
                break
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Tactical training epoch {epoch+1}, avg loss: {avg_loss:.4f}")
    
    return total_loss / batch_count if batch_count > 0 else 0


# Main execution with self-play integration
# Main execution with professional games first, then regular games
if __name__ == "__main__":
    # Initialize model and basic setup - unchanged
    model = ChessNet(num_blocks=10, channels=256).to(device)  
    save_path = "./chess_model/chess_model.pth"
    state_file = "./chess_model/training_state.json"
    pro_state_file = "./chess_model/pro_training_state.json"
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded existing model from {save_path}")

    # Get training state
    current_epoch = 0
    processed_games = 0
    pro_game_count = 0
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_games = state.get("processed_games", 0)
            current_epoch = state.get("last_epoch", 0)
    
    if os.path.exists(pro_state_file):
        with open(pro_state_file, 'r') as f:
            pro_state = json.load(f)
            pro_files_remaining = pro_state.get("current_pro_file_idx", 0) < len(glob.glob(os.path.join("./chess_pgns/pros", "*.pgn")))
            pro_game_count = pro_state.get("processed_pro_games", 0)
            
            # If we've already processed all pro files, start with regular games
            if not pro_files_remaining:
                print("All professional games have been processed. Starting with regular games.")
                current_phase = "regular"
            else:
                current_phase = "professional"
    else:
        current_phase = "professional"  # Start with professional games by default
        pro_game_count = 0

    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))

    # Get file paths
    pgn_directory = "./chess_pgns"
    pgn_files = glob.glob(os.path.join(pgn_directory, "*.pgn"))

    # Load puzzles once - they're small enough
    puzzle_pgn = "./chess_pgns/puzzles/puzzles.pgn"
    lichess_csv = "./chess_pgns/puzzles/lichess_db_puzzle.csv"
    
    # Only try to load if files exist
    pgn_puzzles = []
    if os.path.exists(puzzle_pgn):
        pgn_puzzles = load_puzzles(puzzle_pgn)
        print(f"Loaded {len(pgn_puzzles)} PGN puzzles")
    
    lichess_puzzles = []  
    if os.path.exists(lichess_csv):
        lichess_puzzles = load_lichess_puzzles(lichess_csv)
        print(f"Loaded {len(lichess_puzzles)} Lichess puzzles")

    all_puzzles = pgn_puzzles + lichess_puzzles
    prioritized_puzzles = filter_and_prioritize_puzzles_cached(all_puzzles)
    puzzle_dataset = PuzzleDataset(prioritized_puzzles)
    print(f"Total puzzles after prioritization: {len(prioritized_puzzles)}")

    # Find optimal batch size
    print("Determining optimal batch size...")
    model.eval()
    optimal_batch_size = get_optimal_batch_size(starting_size=64)
    model.train()
    print(f"Using optimal batch size: {optimal_batch_size}")
    
    # Create puzzle dataloader once - puzzles are smaller and reused
    puzzle_dataloader = DataLoader(
        puzzle_dataset, 
        batch_size=min(32, optimal_batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Get current training phase from state file or determine by counts
    if pro_game_count < 1000000:  # Arbitrary threshold for switching to regular games
        current_phase = "professional"
    else:
        current_phase = "regular"
    
    # Command-line override option for phase
    if len(sys.argv) > 1 and sys.argv[1] in ["pro", "regular"]:
        current_phase = "professional" if sys.argv[1] == "pro" else "regular"
        print(f"Command-line override: Using {current_phase} phase")
    
    # Set batch sizes - smaller batch sizes for faster processing
    pro_batch_size = 1000
    regular_batch_size = 1000
    
    # Main training loop
    max_iterations = 1000  # Safety limit
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        print(f"\n--- Training Iteration {iterations} ---")
        
        if current_phase == "professional":
            print("\n=== PROFESSIONAL GAMES TRAINING ===")
            
            # Load one batch of professional games
            pro_games = load_professional_games(pro_state_file, batch_size=pro_batch_size)
            
            if not pro_games:
                print("No more professional games to process. Permanently switching to regular games.")
                # Mark in the state file that we've processed all pro games
                with open(pro_state_file, 'w') as f:
                    pro_state = {"processed_pro_games": pro_game_count,
                                "all_pro_games_processed": True}
                    json.dump(pro_state, f)
                current_phase = "regular"
                continue

            
            batch_size = len(pro_games)
            print(f"Processing professional batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(pro_games, augment=True)
            game_dataloader = DataLoader(
                game_dataset, 
                batch_size=optimal_batch_size,
                shuffle=True, 
                num_workers=min(2, os.cpu_count() or 1),
                pin_memory=True
            )
            
            # Train on this batch
            train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, 
                    epochs=5, processed_games=processed_games)
            
            # Clean up to free memory before next phase
            del pro_games
            del game_dataset
            del game_dataloader
            gc.collect()
            clear_memory()
            
            # Tactical training after professional batch
            print("Running quick tactical training phase...")
            tactical_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
            train_tactical(model, tactical_optimizer, puzzle_dataloader, device, epochs=3)
            
            # Generate a few self-play games after each pro batch
            self_play_count = min(iterations, 5)
            print(f"Generating {self_play_count} self-play games...")
            self_play_games = generate_self_play_games(model, num_games=self_play_count)
            
            if self_play_games:
                self_play_dataset = ChessDataset(self_play_games, augment=True)
                self_play_dataloader = DataLoader(
                    self_play_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True
                )
                train_batch(model, self_play_dataloader, puzzle_dataloader, save_path, pro_state_file, 
                        epochs=1, processed_games=processed_games)
                
                del self_play_games
                del self_play_dataset
                del self_play_dataloader
                gc.collect()
                clear_memory()
            
            # Determine if we should move to regular phase (this can be tuned)
            if pro_game_count > 10000 or iterations % 5 == 0:
                current_phase = "regular"
                
        else:  # Regular games phase
            print("\n=== REGULAR GAMES TRAINING ===")
            
            # Load one batch of regular games
            regular_games = load_games_in_batches(pgn_files, state_file, batch_size=regular_batch_size)
            
            if not regular_games:
                print("No regular games available or error loading games.")
                current_phase = "professional"  # Switch back to pro games if issues with regular games
                continue
                
            batch_size = len(regular_games)
            print(f"Processing regular batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(regular_games, augment=True)
            game_dataloader = DataLoader(
                game_dataset, 
                batch_size=optimal_batch_size,
                shuffle=True, 
                num_workers=min(2, os.cpu_count() or 1),
                pin_memory=True
            )
            
            # Train on regular games
            train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, 
                    epochs=5, processed_games=processed_games)
            
            # Clean up
            del regular_games
            del game_dataset
            del game_dataloader
            gc.collect()
            clear_memory()
            
            # Generate some self-play games after regular batch
            self_play_count = min(5 + iterations // 2, 10)
            print(f"Generating {self_play_count} self-play games...")
            self_play_games = generate_self_play_games(model, num_games=self_play_count)
            
            if self_play_games:
                self_play_dataset = ChessDataset(self_play_games, augment=True)
                self_play_dataloader = DataLoader(
                    self_play_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True
                )
                train_batch(model, self_play_dataloader, puzzle_dataloader, save_path, state_file, 
                        epochs=1, processed_games=processed_games)
                
                del self_play_games
                del self_play_dataset
                del self_play_dataloader
                gc.collect()
                clear_memory()
            
            # Run tactical test occasionally
            if iterations % 3 == 0:
                test_accuracy = test_tactical_recognition(model, device)
                print(f"Tactical recognition accuracy: {test_accuracy:.2%}")
            
            # Switch back to pro games every few iterations
            if iterations % 3 == 0 and not pro_state.get("all_pro_games_processed", False):
                current_phase = "professional"
        
        # Save checkpoint every iteration
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint (iteration {iterations})")
        
        # Update tracking
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_games = state.get("processed_games", 0)
        
        with open(pro_state_file, 'r') as f:
            pro_state = json.load(f)
            pro_game_count = pro_state.get("processed_pro_games", 0)
        
        print(f"Progress: {pro_game_count} professional games, {processed_games} regular games")
        
        # Give user a chance to interrupt gracefully
        print("Waiting 5 seconds before next iteration (Ctrl+C to stop)...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None, model, save_path, state_file, processed_games, current_epoch)
            break
    
    print(f"\nTraining completed with {processed_games} regular games and {pro_game_count} professional games!")