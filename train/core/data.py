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
import time
from functools import lru_cache
from pathlib import Path
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler

from .constants import promotion_moves

def get_move_index(move):
    """Get the index of a move in the policy vector"""
    if move.promotion:
        return promotion_moves[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square


@lru_cache(maxsize=4096)
def _board_to_tensor_cached(fen: str, move_number: int, input_channels: int):
    """Cached inner implementation: FEN + scalars -> tensor."""
    board = chess.Board(fen)
    # Only 18, 20, or 22 channels supported
    if input_channels not in [18, 20, 22]:
        raise ValueError(f"Unsupported input_channels: {input_channels}. Use 18, 20, or 22.")

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


def board_to_tensor(board, move_number=None, input_channels=22):
    """Convert a chess board to a tensor representation (LRU-cached by FEN)."""
    if move_number is None:
        move_number = board.fullmove_number
    return _board_to_tensor_cached(board.fen(), move_number, input_channels)


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
    """
    Optimized puzzle dataset with disk-based tensor caching.
    
    Pre-computes all tensors ONCE and saves to disk cache.
    Subsequent runs load from cache instantly with minimal RAM usage.
    Uses numpy memmap for memory-efficient disk access.
    """
    def __init__(self, puzzles, model_type="big", cache_dir="./cache"):
        self.puzzles = puzzles
        self.model_type = model_type
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # Channel configuration
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:
            self.input_channels = 22
        
        # Generate cache key based on puzzle count and model type
        cache_key = hashlib.md5(f"{len(puzzles)}_{model_type}_{self.input_channels}".encode()).hexdigest()[:12]
        self.tensor_cache_file = os.path.join(cache_dir, f"puzzle_tensors_{cache_key}.npy")
        self.policy_cache_file = os.path.join(cache_dir, f"puzzle_policies_{cache_key}.npy")
        self.value_cache_file = os.path.join(cache_dir, f"puzzle_values_{cache_key}.npy")
        self.category_cache_file = os.path.join(cache_dir, f"puzzle_categories_{cache_key}.pkl")
        
        # Load or create cache
        if self._cache_exists():
            self._load_cache()
        else:
            self._create_cache()
    
    def _cache_exists(self):
        return (os.path.exists(self.tensor_cache_file) and 
                os.path.exists(self.policy_cache_file) and
                os.path.exists(self.value_cache_file) and
                os.path.exists(self.category_cache_file))
    
    def _load_cache(self):
        """Load pre-computed tensors from disk cache (memory-mapped for low RAM)."""
        print(f"Loading cached tensors from {self.cache_dir}...")
        
        # Memory-map the tensor file (doesn't load into RAM until accessed)
        self.tensors = np.load(self.tensor_cache_file, mmap_mode='r')
        self.policies = np.load(self.policy_cache_file, mmap_mode='r')
        self.values = np.load(self.value_cache_file, mmap_mode='r')
        
        with open(self.category_cache_file, 'rb') as f:
            self.categories = pickle.load(f)
        
        print(f"✓ Loaded {len(self.tensors)} cached puzzles (memory-mapped)")
    
    def _create_cache(self):
        """Pre-compute all tensors and save to disk cache."""
        n_puzzles = len(self.puzzles)
        print(f"Pre-computing {n_puzzles} puzzle tensors (one-time operation)...")
        print("This will be cached for instant loading next time.")
        
        # Pre-allocate arrays
        tensors = np.zeros((n_puzzles, self.input_channels, 8, 8), dtype=np.float32)
        policies = np.zeros(n_puzzles, dtype=np.int64)
        values = np.zeros(n_puzzles, dtype=np.float32)
        categories = []
        
        # Process in batches to show progress
        batch_size = 1000
        start_time = time.time()
        
        for i in range(0, n_puzzles, batch_size):
            batch_end = min(i + batch_size, n_puzzles)
            
            for j in range(i, batch_end):
                puzzle = self.puzzles[j]
                
                # Handle both 3-tuple and 4-tuple formats
                if len(puzzle) == 4:
                    fen, move_uci, value_target, category = puzzle
                else:
                    fen, move_uci, value_target = puzzle
                    category = "other"
                
                try:
                    board = chess.Board(fen)
                    move = chess.Move.from_uci(move_uci)
                    
                    tensors[j] = board_to_tensor(board, 0, self.input_channels)
                    policies[j] = get_move_index(move)
                    values[j] = value_target
                    categories.append(category)
                except Exception as e:
                    # Use zeros for invalid puzzles
                    categories.append("other")
            
            # Progress update
            elapsed = time.time() - start_time
            rate = (i + batch_size) / elapsed if elapsed > 0 else 0
            eta = (n_puzzles - i - batch_size) / rate if rate > 0 else 0
            print(f"  {batch_end}/{n_puzzles} ({100*batch_end/n_puzzles:.1f}%) - {rate:.0f} puzzles/sec - ETA: {eta:.0f}s")
        
        # Save to disk
        print("Saving to cache...")
        np.save(self.tensor_cache_file, tensors)
        np.save(self.policy_cache_file, policies)
        np.save(self.value_cache_file, values)
        with open(self.category_cache_file, 'wb') as f:
            pickle.dump(categories, f)
        
        # Load as memory-mapped
        self.tensors = np.load(self.tensor_cache_file, mmap_mode='r')
        self.policies = np.load(self.policy_cache_file, mmap_mode='r')
        self.values = np.load(self.value_cache_file, mmap_mode='r')
        self.categories = categories
        
        total_time = time.time() - start_time
        print(f"✓ Cached {n_puzzles} puzzles in {total_time:.1f}s")

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        # Fast access from cache - no computation needed!
        return (torch.from_numpy(self.tensors[idx].copy()),
                torch.tensor(self.policies[idx], dtype=torch.long),
                torch.tensor(self.values[idx], dtype=torch.float32),
                self.categories[idx])


class CurriculumPuzzleDataset(Dataset):
    """Puzzle dataset with curriculum learning - progressive difficulty stages.
    
    Implements staged training where the model starts with easy puzzles
    (mate-in-1) and progressively trains on harder puzzles as it improves.
    
    Stages:
    1. mate_basics: Only mate-in-1 puzzles (easiest pattern recognition)
    2. mate_extended: All mate puzzles (mate-in-1, 2, 3, longer)
    3. tactics: Mates + tactical puzzles (forks, pins, etc.)
    4. full: Complete puzzle set
    
    The curriculum advances automatically when accuracy threshold is met,
    or can be manually advanced via advance_stage().
    """
    
    # Stage definitions: name, categories to include, accuracy threshold to advance
    STAGES = [
        {
            "name": "mate_basics",
            "categories": {"mate_in_one"},
            "threshold": 0.70,
            "description": "Mate-in-1 only (pattern recognition)"
        },
        {
            "name": "mate_extended", 
            "categories": {"mate_in_one", "mate_in_two", "mate_in_three", 
                          "mate_longer", "backrank_mate", "smothered_mate"},
            "threshold": 0.60,
            "description": "All mate puzzles"
        },
        {
            "name": "tactics",
            "categories": {"mate_in_one", "mate_in_two", "mate_in_three",
                          "mate_longer", "backrank_mate", "smothered_mate",
                          "fork", "pin", "skewer", "discovered", "double_attack"},
            "threshold": 0.50,
            "description": "Mates + tactical puzzles"
        },
        {
            "name": "full",
            "categories": None,  # None means all categories
            "threshold": None,   # No advancement (final stage)
            "description": "Complete puzzle set"
        }
    ]
    
    def __init__(self, puzzles, model_type="big", cache_dir="./cache", start_stage=0):
        """Initialize curriculum dataset.
        
        Args:
            puzzles: List of puzzle tuples (fen, move, value, category)
            model_type: Model size for tensor generation
            cache_dir: Directory for caching tensors
            start_stage: Initial curriculum stage (0-3)
        """
        self.all_puzzles = puzzles
        self.model_type = model_type
        self.cache_dir = cache_dir
        self.current_stage = min(start_stage, len(self.STAGES) - 1)
        
        # Channel configuration
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:
            self.input_channels = 22
        
        # Build category -> puzzle indices mapping
        self._build_category_index()
        
        # Set initial stage
        self._update_active_puzzles()
        
        print(f"Curriculum initialized at stage {self.current_stage}: {self.STAGES[self.current_stage]['name']}")
        print(f"  {self.STAGES[self.current_stage]['description']}")
        print(f"  Active puzzles: {len(self.active_indices)}")
    
    def _build_category_index(self):
        """Build index mapping categories to puzzle indices."""
        self.category_indices = {}
        
        for i, puzzle in enumerate(self.all_puzzles):
            category = puzzle[3] if len(puzzle) >= 4 else "other"
            if category not in self.category_indices:
                self.category_indices[category] = []
            self.category_indices[category].append(i)
        
        # Print category summary
        print("Curriculum puzzle categories:")
        for cat, indices in sorted(self.category_indices.items(), 
                                   key=lambda x: -len(x[1]))[:8]:
            print(f"  {cat}: {len(indices)}")
    
    def _update_active_puzzles(self):
        """Update active puzzle indices based on current stage."""
        stage = self.STAGES[self.current_stage]
        allowed_categories = stage["categories"]
        
        if allowed_categories is None:
            # Full stage - use all puzzles
            self.active_indices = list(range(len(self.all_puzzles)))
        else:
            # Filter to allowed categories
            self.active_indices = []
            for category in allowed_categories:
                if category in self.category_indices:
                    self.active_indices.extend(self.category_indices[category])
        
        # Shuffle for better training
        import random
        random.shuffle(self.active_indices)
    
    def advance_stage(self):
        """Advance to next curriculum stage.
        
        Returns:
            bool: True if advanced, False if already at final stage
        """
        if self.current_stage >= len(self.STAGES) - 1:
            return False
        
        self.current_stage += 1
        self._update_active_puzzles()
        
        stage = self.STAGES[self.current_stage]
        print(f"\n🎓 CURRICULUM ADVANCED to stage {self.current_stage}: {stage['name']}")
        print(f"   {stage['description']}")
        print(f"   Active puzzles: {len(self.active_indices)}")
        
        return True
    
    def check_and_advance(self, accuracy):
        """Check if accuracy meets threshold and advance if so.
        
        Args:
            accuracy: Current accuracy (0.0-1.0)
            
        Returns:
            bool: True if stage was advanced
        """
        stage = self.STAGES[self.current_stage]
        threshold = stage.get("threshold")
        
        if threshold is not None and accuracy >= threshold:
            print(f"✓ Accuracy {accuracy:.2%} meets threshold {threshold:.0%}")
            return self.advance_stage()
        
        return False
    
    def get_stage_info(self):
        """Get information about current stage."""
        stage = self.STAGES[self.current_stage]
        return {
            "index": self.current_stage,
            "name": stage["name"],
            "description": stage["description"],
            "num_puzzles": len(self.active_indices),
            "threshold": stage.get("threshold"),
            "is_final": self.current_stage >= len(self.STAGES) - 1
        }
    
    def __len__(self):
        return len(self.active_indices)
    
    def __getitem__(self, idx):
        """Get puzzle at index (from active set only).
        
        Applies random augmentation (horizontal mirroring) 50% of the time
        to increase effective training data diversity and reduce overfitting.
        """
        puzzle_idx = self.active_indices[idx]
        puzzle = self.all_puzzles[puzzle_idx]
        
        # Handle 3-tuple and 4-tuple formats
        if len(puzzle) >= 4:
            fen, move_uci, value_target, category = puzzle[:4]
        else:
            fen, move_uci, value_target = puzzle[:3]
            category = "other"
        
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(move_uci)
            
            # Random augmentation: horizontal mirror 50% of the time
            # This doubles effective training data and improves generalization
            import random
            if random.random() < 0.5:
                # Mirror the board horizontally (flip a-h to h-a)
                board = board.mirror()
                move = chess.Move(
                    chess.square_mirror(move.from_square),
                    chess.square_mirror(move.to_square),
                    move.promotion
                )
            
            
            # Only 18/20/22 channels supported
            input_tensor = board_to_tensor(board, 0, self.input_channels)
            policy_target = get_move_index(move)
            
            return (torch.tensor(input_tensor, dtype=torch.float32),
                    torch.tensor(policy_target, dtype=torch.long),
                    torch.tensor(value_target, dtype=torch.float32),
                    category)
        except Exception as e:
            # Return dummy data on error (will be rare)
            return (torch.zeros(self.input_channels, 8, 8),
                    torch.tensor(0, dtype=torch.long),
                    torch.tensor(0.0, dtype=torch.float32),
                    "error")


class ProGameDataset(Dataset):
    """Dataset wrapping positions from labelled pro PGNs.

    Stores compact (fen, move_uci, value_target) tuples; mirrors the board
    horizontally for augmentation the same way ChessDataset does.
    """

    def __init__(self, samples, augment=True, model_type="big"):
        # samples: list of (fen, move_uci, value_target)
        self.augment = augment
        self.model_type = model_type
        model_lower = model_type.lower()
        if model_lower in ["small", "limited"]:
            self.input_channels = 18
        elif model_lower == "medium":
            self.input_channels = 20
        else:
            self.input_channels = 22
        self.positions = []
        for sample in samples:
            fen, move_uci, value_target = sample[:3]
            try:
                move = chess.Move.from_uci(move_uci)
                policy_target = get_move_index(move)
            except Exception:
                continue
            self.positions.append((fen, policy_target, float(value_target)))
            if self.augment:
                try:
                    mirrored_board = chess.Board(fen).mirror()
                    mirrored_move = chess.Move(
                        chess.square_mirror(move.from_square),
                        chess.square_mirror(move.to_square),
                        move.promotion,
                    )
                    mirrored_policy = get_move_index(mirrored_move)
                    self.positions.append((mirrored_board.fen(), mirrored_policy, float(value_target)))
                except Exception:
                    pass

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        fen, policy_target, value_target = self.positions[idx]
        board = chess.Board(fen)
        input_tensor = board_to_tensor(board, 0, self.input_channels)
        return (
            torch.tensor(input_tensor, dtype=torch.float32),
            torch.tensor(policy_target, dtype=torch.long),
            torch.tensor(value_target, dtype=torch.float32),
        )


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
        
        if isinstance(policy_target, (int, np.integer)):
            policy_tensor = torch.tensor(policy_target, dtype=torch.long)
        else:
            policy_tensor = torch.tensor(policy_target, dtype=torch.float32)

        return (torch.tensor(board_tensor, dtype=torch.float32),
                policy_tensor,
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

def load_lichess_puzzles(csv_file, cache_dir="./cache"):
    """Load puzzles from Lichess CSV file with category extraction.
    
    Extracts puzzle type from the 'Themes' column.
    Lichess themes include: mateIn1, mateIn2, fork, pin, skewer, etc.
    
    Uses file-based caching to skip CSV parsing on subsequent runs.
    
    Returns:
        List of tuples. For mate puzzles, returns 5-tuple:
        (fen, first_move_uci, value_target, category, full_moves_list)
        For non-mate puzzles, returns 4-tuple:
        (fen, move_uci, value_target, category)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on file path and modification time
    file_stat = os.stat(csv_file)
    file_hash = hashlib.md5(f"{csv_file}_{file_stat.st_size}_{file_stat.st_mtime}".encode()).hexdigest()[:12]
    cache_file = os.path.join(cache_dir, f"lichess_puzzles_{file_hash}.pkl")
    
    # Try loading from cache
    if os.path.exists(cache_file):
        try:
            print(f"Loading Lichess puzzles from cache...")
            with open(cache_file, 'rb') as f:
                puzzles = pickle.load(f)
            print(f"Loaded {len(puzzles)} Lichess puzzles from cache (instant!)")
            return puzzles
        except Exception as e:
            print(f"Cache error: {e}, rebuilding...")
    
    print(f"Parsing Lichess CSV (first time, will cache)...")
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
            is_mate_puzzle = False
            if 'matein1' in themes or 'mate in 1' in themes:
                category = "mate_in_one"
                value_target = 1.0
                is_mate_puzzle = True
            elif 'matein2' in themes or 'mate in 2' in themes:
                category = "mate_in_two"
                value_target = 1.0
                is_mate_puzzle = True
            elif 'matein3' in themes or 'mate in 3' in themes:
                category = "mate_in_three"
                value_target = 1.0
                is_mate_puzzle = True
            elif 'matein4' in themes or 'matein5' in themes or 'mate' in themes:
                category = "mate_longer"
                value_target = 1.0
                is_mate_puzzle = True
            elif 'backrankmatepattern' in themes or 'backrankmatemate' in themes:
                category = "backrank_mate"
                value_target = 1.0
                is_mate_puzzle = True
            elif 'smotheredmate' in themes:
                category = "smothered_mate"
                value_target = 1.0
                is_mate_puzzle = True
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
            
            # For mate puzzles, store the full move sequence (5-tuple)
            # This allows us to expand into intermediate positions later
            if is_mate_puzzle and len(moves) > 1:
                puzzles.append((fen, move_uci, value_target, category, moves))
            else:
                puzzles.append((fen, move_uci, value_target, category))
            
            category_counts[category] = category_counts.get(category, 0) + 1
    
    # Print category distribution
    print(f"Loaded {len(puzzles)} puzzles from Lichess CSV")
    print("Category distribution:")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cat}: {count}")
    
    # Save to cache for next time
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(puzzles, f)
        print(f"Cached Lichess puzzles to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not cache puzzles: {e}")
    
    return puzzles


def expand_mate_sequences(puzzles, max_expand_depth=4, cache_dir="./cache"):
    """Expand mate-in-N puzzles to include all intermediate positions.
    
    This is crucial for teaching checkmate patterns because the model needs
    to learn the SETUP moves, not just the final checkmate move.
    
    For a mate-in-3 sequence: pos1 -> m1 -> pos2 -> opp1 -> pos3 -> m2 -> pos4 -> opp2 -> pos5 -> m3 (checkmate)
    We generate training samples for positions pos1, pos3, pos5 (our moves only).
    
    Value targets are graduated based on distance to mate:
    - Final checkmate move: 1.0
    - One move before: 0.98
    - Two moves before: 0.95
    - Three moves before: 0.92
    
    Uses caching to avoid re-expanding on every startup.
    
    Args:
        puzzles: List of puzzle tuples (some may be 5-tuples with full move sequences)
        max_expand_depth: Maximum number of positions to expand per puzzle
        cache_dir: Directory for cache files
        
    Returns:
        List of expanded puzzle tuples (fen, move_uci, value_target, category)
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on puzzle count and content hash
    puzzle_hash = hashlib.md5(str(len(puzzles)).encode() + str(puzzles[:10]).encode()).hexdigest()[:12]
    cache_file = os.path.join(cache_dir, f"expanded_mates_{puzzle_hash}.pkl")
    
    # Try loading from cache
    if os.path.exists(cache_file):
        try:
            print(f"Loading expanded mate sequences from cache...")
            with open(cache_file, 'rb') as f:
                expanded = pickle.load(f)
            print(f"Loaded {len(expanded)} expanded puzzles from cache (instant!)")
            return expanded
        except Exception as e:
            print(f"Cache error: {e}, rebuilding...")
    
    print(f"Expanding mate sequences (first time, will cache)...")
    expanded = []
    expanded_count = 0
    
    # Value targets based on distance to checkmate (closer = higher)
    value_by_distance = {
        0: 1.0,    # Final checkmate move
        1: 0.98,   # One move before checkmate
        2: 0.95,   # Two moves before
        3: 0.92,   # Three moves before
        4: 0.88,   # Four moves before
    }
    
    for puzzle in puzzles:
        # Check if this is a mate puzzle with full move sequence (5-tuple)
        if len(puzzle) == 5:
            fen, first_move, base_value, category, moves = puzzle
            
            # Only expand mate puzzles with multiple moves
            if category not in ('mate_in_one', 'mate_in_two', 'mate_in_three', 
                               'mate_longer', 'backrank_mate', 'smothered_mate'):
                # Not a mate puzzle, keep as 4-tuple
                expanded.append((fen, first_move, base_value, category))
                continue
            
            try:
                board = chess.Board(fen)
                positions_generated = 0
                
                # Calculate total moves to checkmate
                # Lichess puzzles: moves[0] is our first move, moves[1] is opponent response, etc.
                our_moves = [(i, moves[i]) for i in range(0, len(moves), 2)]  # Even indices are our moves
                total_our_moves = len(our_moves)
                
                # Generate position for each of our moves
                for move_idx, (seq_idx, move_uci) in enumerate(our_moves):
                    if positions_generated >= max_expand_depth:
                        break
                    
                    # Calculate distance from checkmate (0 = checkmate move)
                    distance_to_mate = total_our_moves - move_idx - 1
                    value_target = value_by_distance.get(distance_to_mate, 0.85)
                    
                    # Determine category based on distance
                    if distance_to_mate == 0:
                        sub_category = "mate_in_one"  # This IS the checkmate move
                    elif distance_to_mate == 1:
                        sub_category = "mate_in_two"  # One move before checkmate
                    else:
                        sub_category = category  # Keep original category
                    
                    # Get current position FEN
                    current_fen = board.fen()
                    
                    # Add this position as a training sample
                    expanded.append((current_fen, move_uci, value_target, sub_category))
                    positions_generated += 1
                    
                    # Apply the move and opponent's response to get to next position
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            
                            # Apply opponent's response if there is one
                            opp_idx = seq_idx + 1
                            if opp_idx < len(moves):
                                opp_move = chess.Move.from_uci(moves[opp_idx])
                                if opp_move in board.legal_moves:
                                    board.push(opp_move)
                        else:
                            break  # Invalid move, stop expanding
                    except:
                        break  # Error, stop expanding
                
                if positions_generated > 1:
                    expanded_count += positions_generated - 1
                    
            except Exception as e:
                # If expansion fails, just add the original puzzle as 4-tuple
                expanded.append((fen, first_move, base_value, category))
        else:
            # Regular 4-tuple or 3-tuple puzzle, keep as-is
            if len(puzzle) == 4:
                expanded.append(puzzle)
            else:  # 3-tuple
                fen, move_uci, value_target = puzzle
                expanded.append((fen, move_uci, value_target, "other"))
    
    print(f"Mate sequence expansion: {len(puzzles)} puzzles -> {len(expanded)} samples (+{expanded_count} intermediate positions)")
    
    # Save to cache for next time
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(expanded, f)
        print(f"Cached expanded puzzles to {cache_file}")
    except Exception as e:
        print(f"Warning: Could not cache expanded puzzles: {e}")
    
    return expanded


def _resolve_chess_pgn_root(root_dir=None):
    """Resolve the chess PGN root directory.

    Defaults to ``train/chess_pgns`` relative to this module.
    """
    if root_dir is not None:
        return Path(root_dir)
    return Path(__file__).resolve().parents[1] / "chess_pgns"


def discover_pgn_files(root_dir=None, subdirs=None):
    """Recursively discover PGN files under the chess PGN root.

    Args:
        root_dir: Root directory (defaults to ``train/chess_pgns``)
        subdirs: Optional iterable of subdirectories to search

    Returns:
        List of absolute PGN file paths as strings
    """
    root = _resolve_chess_pgn_root(root_dir)
    if subdirs is None:
        subdirs = ("pros", "high_elo", "puzzles")

    pgn_files = []
    for subdir in subdirs:
        folder = root / subdir
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*.pgn")):
            pgn_files.append(str(path))
    return pgn_files


def load_pgn_games_from_directory(root_dir=None, subdirs=None, max_games_per_file=None):
    """Load chess.pgn.Game objects from a directory tree.

    This is the supervised training source for pro/high-elo game data.
    """
    import chess.pgn

    games = []
    for pgn_file in discover_pgn_files(root_dir=root_dir, subdirs=subdirs):
        try:
            with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
                loaded = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    games.append(game)
                    loaded += 1
                    if max_games_per_file is not None and loaded >= int(max_games_per_file):
                        break
        except Exception:
            continue
    return games


def load_puzzle_examples_from_directory(root_dir=None, subdirs=None, max_files=None):
    """Load puzzle tuples from PGN files under the chess PGN tree.

    Returns the same tuple shape as :func:`load_puzzles`.
    """
    root = _resolve_chess_pgn_root(root_dir)
    if subdirs is None:
        subdirs = ("puzzles",)

    examples = []
    files = []
    for subdir in subdirs:
        folder = root / subdir
        if not folder.exists():
            continue
        files.extend(sorted(folder.rglob("*.pgn")))

    if max_files is not None:
        files = files[: int(max_files)]

    for path in files:
        try:
            examples.extend(load_puzzles(str(path)))
        except Exception:
            continue
    return examples


def load_training_examples_from_chess_pgns(root_dir=None, include_games=True, include_puzzles=True,
                                          game_subdirs=("pros", "high_elo"), puzzle_subdirs=("puzzles",),
                                          max_games_per_file=None, max_puzzle_files=None):
    """Convenience helper that loads both supervised games and tactical puzzles.

    Returns:
        dict with keys ``games`` and ``puzzles``.
    """
    result = {"games": [], "puzzles": []}
    if include_games:
        result["games"] = load_pgn_games_from_directory(
            root_dir=root_dir,
            subdirs=game_subdirs,
            max_games_per_file=max_games_per_file,
        )
    if include_puzzles:
        result["puzzles"] = load_puzzle_examples_from_directory(
            root_dir=root_dir,
            subdirs=puzzle_subdirs,
            max_files=max_puzzle_files,
        )
    return result


def compute_source_sample_weights(datasets, source_weights=None):
    """Compute per-sample weights for a concatenated dataset mix.

    Each source dataset receives the relative weight specified in
    ``source_weights`` and is normalized by its length so that shorter
    sources are not drowned out by larger ones.

    Args:
        datasets: Sequence of datasets to concatenate.
        source_weights: Optional per-source weights. Defaults to 1.0 each.

    Returns:
        List of per-sample weights aligned with ``ConcatDataset`` ordering.
    """
    datasets = list(datasets)
    if source_weights is None:
        source_weights = [1.0] * len(datasets)

    if len(source_weights) != len(datasets):
        raise ValueError(
            f"source_weights length ({len(source_weights)}) must match datasets length ({len(datasets)})"
        )

    weights = []
    for dataset, source_weight in zip(datasets, source_weights):
        length = len(dataset)
        if length <= 0:
            continue

        per_sample_weight = float(source_weight) / float(length)
        weights.extend([per_sample_weight] * length)

    if not weights:
        raise ValueError("Cannot compute sample weights from an empty dataset mix")

    return weights


def create_balanced_concat_dataloader(
    datasets,
    batch_size,
    source_weights=None,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
    replacement=True,
):
    """Create a balanced DataLoader over multiple datasets.

    This keeps small but important sources like puzzle positions visible during
    training instead of letting the largest dataset dominate every batch.
    """
    datasets = [dataset for dataset in datasets if len(dataset) > 0]
    if not datasets:
        raise ValueError("No non-empty datasets were provided")

    if source_weights is None:
        source_weights = [1.0] * len(datasets)

    if len(source_weights) != len(datasets):
        raise ValueError(
            f"source_weights length ({len(source_weights)}) must match non-empty datasets length ({len(datasets)})"
        )

    weights = compute_source_sample_weights(datasets, source_weights)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=replacement,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(ConcatDataset(datasets), **loader_kwargs)


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
    
    # Limit to reasonable size (smaller dataset = faster training)
    max_puzzles = 100000  # Reduced for faster training
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
