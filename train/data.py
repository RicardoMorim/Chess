import numpy as np
import torch
import chess
import chess.pgn
import csv
import os
import pickle
import hashlib
import random
import glob
from torch.utils.data import Dataset

from constants import promotion_moves

def get_move_index(move):
    """Get the index of a move in the policy vector"""
    if move.promotion:
        return promotion_moves[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square

def board_to_tensor(board, move_number=None, input_channels=22):
    """Convert a chess board to a tensor representation.
    
    This is a critical function that encodes the board state for the neural network.
    
    Channel layout for 22-channel (big) model:
    - 0-5:   White pieces (P, N, B, R, Q, K)
    - 6-11:  Black pieces (p, n, b, r, q, k)
    - 12-15: Castling rights (WK, WQ, BK, BQ)
    - 16:    En passant square
    - 17:    Side to move (1 = white, 0 = black)
    - 18:    Halfmove clock (for 50-move rule)
    - 19:    Fullmove number
    - 20:    White attack map (squares attacked by white)
    - 21:    Black attack map (squares attacked by black)
    
    Channel layout for 18-channel (small) model:
    - 0-5:   White pieces
    - 6-11:  Black pieces
    - 12-15: Castling rights
    - 16:    En passant
    - 17:    Side to move

    Args:
        board: The chess.Board object
        move_number: The move number (only used for 20-channel model)
        input_channels: Number of input channels (18, 20, or 22)
    
    Returns:
        numpy array of shape (input_channels, 8, 8)
    """
    tensor = np.zeros((input_channels, 8, 8), dtype=np.float32)
    
    # Piece positions (channels 0-11)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                channel = piece_type - 1 if color == chess.WHITE else piece_type + 5
                tensor[channel, row, col] = 1.0
    
    # Castling rights (channels 12-15)
    tensor[12, :, :] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[13, :, :] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[14, :, :] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[15, :, :] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # En passant square (channel 16)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[16, row, col] = 1.0
    
    # Side to move (channel 17)
    tensor[17, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Extended features for larger models
    if input_channels >= 20:
        # Halfmove clock normalized (channel 18)
        tensor[18, :, :] = min(board.halfmove_clock / 50.0, 1.0)
        
        # Move number normalized (channel 19)
        move_num = move_number if move_number is not None else board.fullmove_number
        tensor[19, :, :] = min(move_num / 200.0, 1.0)
    
    # Attack maps for 22-channel model
    if input_channels >= 22:
        # White attacks (channel 20)
        for square in chess.SQUARES:
            if board.is_attacked_by(chess.WHITE, square):
                row, col = divmod(square, 8)
                tensor[20, row, col] = 1.0
        
        # Black attacks (channel 21)
        for square in chess.SQUARES:
            if board.is_attacked_by(chess.BLACK, square):
                row, col = divmod(square, 8)
                tensor[21, row, col] = 1.0

    return tensor


class ChessDataset(Dataset):
    """Dataset for chess games with symmetry augmentation and model type support"""
    def __init__(self, games, augment=True, model_type="big"):
        self.positions = []
        self.augment = augment
        self.model_type = model_type
        # Channel configuration: small/limited=18, medium=20, big=22
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:  # big
            self.input_channels = 22
        
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
        # Use the appropriate tensor representation based on model type
        input_tensor = board_to_tensor(board, move_number, self.input_channels)
        
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

class PuzzleDataset(Dataset):
    """Dataset for chess puzzles with category support for weighted training."""
    def __init__(self, puzzles, model_type="big"):
        self.puzzles = puzzles
        self.model_type = model_type
        # Channel configuration: small/limited=18, medium=20, big=22
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:  # big
            self.input_channels = 22

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        puzzle = self.puzzles[idx]
        
        # Handle both 3-tuple (legacy) and 4-tuple (with category) formats
        if len(puzzle) == 4:
            fen, move_uci, value_target, category = puzzle
        else:
            fen, move_uci, value_target = puzzle
            category = "other"  # Default category for legacy puzzles
        
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        # Use the appropriate tensor representation based on model type
        input_tensor = board_to_tensor(board, 0, self.input_channels)
        policy_target = get_move_index(move)
        
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32),
                category)  # Return category for weighted training

class SelfPlayDataset(Dataset):
    """Dataset for self-play reinforcement learning samples with model type support"""
    def __init__(self, samples, model_type="big"):
        self.samples = samples
        self.model_type = model_type
        # Channel configuration: small/limited=18, medium=20, big=22
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:  # big
            self.input_channels = 22

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        board_tensor, policy_target, value_target = self.samples[idx]
        
        # Make sure the tensor has the right dimensions for the model type
        if isinstance(board_tensor, np.ndarray) and board_tensor.shape[0] != self.input_channels:
            print(f"Warning: Converting tensor from {board_tensor.shape[0]} to {self.input_channels} channels")
            if self.input_channels == 18 and board_tensor.shape[0] > 18:
                # Convert from 20/22 to 18 channels by trimming
                board_tensor = board_tensor[:18]
            elif self.input_channels == 22 and board_tensor.shape[0] < 22:
                # Convert from 18/20 to 22 channels by padding
                padded = np.zeros((22, 8, 8), dtype=np.float32)
                padded[:board_tensor.shape[0]] = board_tensor
                # Note: Attack maps (channels 20-21) will be zeros - ideally regenerate from board
                board_tensor = padded
        
        return (torch.tensor(board_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

def load_puzzles(pgn_file):
    """Load puzzles from PGN file with category extraction.
    
    Extracts puzzle type from the 'White' header tag.
    Examples: 'Mate in one', 'Mate in two', 'Fork', 'Pin', etc.
    
    Returns:
        List of (fen, move_uci, value_target, category) tuples
    """
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
                
                # Extract category from White header
                white_header = game.headers.get("White", "").lower()
                
                # Determine category and value based on puzzle type
                if "mate in one" in white_header or "mate in 1" in white_header:
                    category = "mate_in_one"
                    value_target = 1.0
                elif "mate in two" in white_header or "mate in 2" in white_header:
                    category = "mate_in_two"
                    value_target = 1.0
                elif "mate in three" in white_header or "mate in 3" in white_header:
                    category = "mate_in_three"
                    value_target = 1.0
                elif "mate" in white_header:
                    category = "mate_longer"
                    value_target = 1.0
                elif "fork" in white_header:
                    category = "fork"
                    value_target = 0.9
                elif "pin" in white_header:
                    category = "pin"
                    value_target = 0.85
                elif "skewer" in white_header:
                    category = "skewer"
                    value_target = 0.85
                elif "discovered" in white_header:
                    category = "discovered"
                    value_target = 0.85
                elif "double" in white_header:
                    category = "double_attack"
                    value_target = 0.85
                elif "endgame" in white_header or "ending" in white_header:
                    category = "endgame"
                    value_target = 0.8
                else:
                    category = "other"
                    value_target = 0.7
                
                puzzles.append((fen, move_uci, value_target, category))
            except IndexError:
                continue
    
    print(f"Loaded {len(puzzles)} puzzles from PGN")
    return puzzles

def load_lichess_puzzles(csv_file):
    """Load puzzles from Lichess CSV file with category extraction.
    
    Extracts puzzle type from the 'Themes' column.
    Lichess themes include: mateIn1, mateIn2, fork, pin, skewer, etc.
    
    Returns:
        List of (fen, move_uci, value_target, category) tuples
    """
    puzzles = []
    category_counts = {}
    
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row['FEN']
            moves = row['Moves'].split()
            if not moves:
                continue
                
            move_uci = moves[0]
            themes = row.get('Themes', '').lower()
            
            # Determine category from Lichess themes (prioritize mate puzzles)
            if 'matein1' in themes or 'mate in 1' in themes:
                category = "mate_in_one"
                value_target = 1.0
            elif 'matein2' in themes or 'mate in 2' in themes:
                category = "mate_in_two"
                value_target = 1.0
            elif 'matein3' in themes or 'mate in 3' in themes:
                category = "mate_in_three"
                value_target = 1.0
            elif 'matein4' in themes or 'matein5' in themes or 'mate' in themes:
                category = "mate_longer"
                value_target = 1.0
            elif 'backrankmatepattern' in themes or 'backrankmatemate' in themes:
                category = "backrank_mate"
                value_target = 1.0
            elif 'smotheredmate' in themes:
                category = "smothered_mate"
                value_target = 1.0
            elif 'fork' in themes or 'doubleatack' in themes:
                category = "fork"
                value_target = 0.9
            elif 'pin' in themes:
                category = "pin"
                value_target = 0.85
            elif 'skewer' in themes:
                category = "skewer"
                value_target = 0.85
            elif 'discoveredattack' in themes:
                category = "discovered"
                value_target = 0.85
            elif 'endgame' in themes:
                category = "endgame"
                value_target = 0.8
            elif 'promotion' in themes or 'queening' in themes:
                category = "promotion"
                value_target = 0.85
            elif 'sacrifice' in themes:
                category = "sacrifice"
                value_target = 0.8
            else:
                category = "other"
                value_target = 0.7
            
            puzzles.append((fen, move_uci, value_target, category))
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Print category distribution
    print(f"Loaded {len(puzzles)} puzzles from Lichess CSV")
    print("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    return puzzles

def filter_and_prioritize_puzzles_cached(puzzles, cache_dir="./cache"):
    """Filter and prioritize puzzles, with heavy emphasis on mate puzzles.
    
    Properly handles both 3-tuple (legacy) and 4-tuple (with category) formats.
    Prioritizes mate puzzles for checkmate pattern learning.
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a hash based on the puzzles to use as cache key
    puzzles_hash = hashlib.md5(str(len(puzzles)).encode()).hexdigest()[:10]
    cache_file = os.path.join(cache_dir, f"puzzle_cache_v2_{puzzles_hash}.pkl")
    
    # If cache exists, load from it
    if os.path.exists(cache_file):
        print(f"Loading prioritized puzzles from cache...")
        try:
            with open(cache_file, 'rb') as f:
                prioritized_puzzles = pickle.load(f)
            
            print(f"Loaded {len(prioritized_puzzles)} puzzles from cache")
            return prioritized_puzzles
        except Exception as e:
            print(f"Error loading cache: {e}. Rebuilding...")
    
    print(f"Prioritizing {len(puzzles)} puzzles (this may take a while)...")
    
    # Categorize puzzles
    mate_in_one = []
    mate_in_two = []
    mate_in_three = []
    mate_longer = []
    fork_puzzles = []
    pin_puzzles = []
    endgame_puzzles = []
    other_puzzles = []
    
    # Process puzzles
    for puzzle in puzzles:
        # Handle both 3-tuple and 4-tuple formats
        if len(puzzle) == 4:
            fen, move_uci, value_target, category = puzzle
        else:
            fen, move_uci, value_target = puzzle
            # Try to detect category from position
            try:
                board = chess.Board(fen)
                move = chess.Move.from_uci(move_uci)
                test_board = board.copy()
                test_board.push(move)
                if test_board.is_checkmate():
                    category = "mate_in_one"
                elif detect_fork(board, move):
                    category = "fork"
                elif detect_pin(board, move):
                    category = "pin"
                else:
                    category = "other"
            except:
                category = "other"
        
        # Create 4-tuple for consistency
        puzzle_4 = (fen, move_uci, value_target, category)
        
        # Sort into category buckets
        if category == "mate_in_one":
            mate_in_one.append(puzzle_4)
        elif category == "mate_in_two":
            mate_in_two.append(puzzle_4)
        elif category == "mate_in_three":
            mate_in_three.append(puzzle_4)
        elif category in ["mate_longer", "backrank_mate", "smothered_mate"]:
            mate_longer.append(puzzle_4)
        elif category == "fork":
            fork_puzzles.append(puzzle_4)
        elif category in ["pin", "skewer", "discovered"]:
            pin_puzzles.append(puzzle_4)
        elif category == "endgame":
            endgame_puzzles.append(puzzle_4)
        else:
            other_puzzles.append(puzzle_4)
    
    # Combine with heavy priority on mate puzzles (the user's model needs this!)
    prioritized_puzzles = (
        mate_in_one * 10 +      # Mate-in-1: 10x (most important for learning checkmates)
        mate_in_two * 8 +       # Mate-in-2: 8x
        mate_in_three * 6 +     # Mate-in-3: 6x
        mate_longer * 4 +       # Longer mates: 4x
        endgame_puzzles * 4 +   # Endgames: 4x (important for finishing)
        fork_puzzles * 2 +      # Forks: 2x
        pin_puzzles * 2 +       # Pins: 2x
        other_puzzles           # Others: 1x
    )
    
    # Shuffle to mix categories
    random.shuffle(prioritized_puzzles)
    
    # Limit to reasonable size
    max_puzzles = 300000
    if len(prioritized_puzzles) > max_puzzles:
        print(f"Too many puzzles ({len(prioritized_puzzles)}), sampling {max_puzzles}")
        prioritized_puzzles = prioritized_puzzles[:max_puzzles]
    
    # Print statistics
    print(f"\nPrioritized puzzles: {len(prioritized_puzzles)} (from {len(puzzles)} original)")
    print(f"  Mate-in-1: {len(mate_in_one)} (x10 = {len(mate_in_one)*10})")
    print(f"  Mate-in-2: {len(mate_in_two)} (x8 = {len(mate_in_two)*8})")
    print(f"  Mate-in-3: {len(mate_in_three)} (x6 = {len(mate_in_three)*6})")
    print(f"  Longer mates: {len(mate_longer)} (x4 = {len(mate_longer)*4})")
    print(f"  Endgames: {len(endgame_puzzles)} (x4 = {len(endgame_puzzles)*4})")
    print(f"  Forks: {len(fork_puzzles)} (x2)")
    print(f"  Pins: {len(pin_puzzles)} (x2)")
    print(f"  Others: {len(other_puzzles)}")
    
    # Cache the results
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(prioritized_puzzles, f)
        print(f"Cached to {cache_file}")
    except Exception as e:
        print(f"Error caching: {e}")
    
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
