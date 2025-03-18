import numpy as np
import torch
import chess
import chess.pgn
import csv
import os
import pickle
import hashlib
import random
from torch.utils.data import Dataset

from constants import promotion_moves

def get_move_index(move):
    """Get the index of a move in the policy vector"""
    if move.promotion:
        return promotion_moves[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square

def board_to_tensor(board, move_number):
    """Convert a chess board to a tensor representation"""
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

class ChessDataset(Dataset):
    """Dataset for chess games with symmetry augmentation"""
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

class PuzzleDataset(Dataset):
    """Dataset for chess puzzles"""
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

class SelfPlayDataset(Dataset):
    """Dataset for self-play reinforcement learning samples"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board_tensor, policy_target, value_target = self.samples[idx]
        return (torch.tensor(board_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

def load_puzzles(pgn_file):
    """Load puzzles from PGN file"""
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
    """Load puzzles from Lichess CSV file"""
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

def load_professional_games(state_file, batch_size=1500, max_games=1500):
    """Load professional games more efficiently - one file at a time"""
    import glob
    import json
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

def load_games_in_batches(pgn_files, state_file, batch_size=1500):
    """Load regular games more efficiently - file position tracking"""
    import json
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
