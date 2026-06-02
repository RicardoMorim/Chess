"""
Core Chess AI Components
========================

This module contains the core components used across all training systems:
- models: Neural network architectures (ChessNet, ESTNet)
- mcts: Monte Carlo Tree Search implementation
- training: Training utilities, losses, optimizers
- data: Board representation and move encoding
- constants: Configuration constants

Import from here for consistent access:
    from core import create_model, MCTS, board_to_tensor
    from core.models import ChessNet
    from core.data import get_move_index
"""

# Models
from .models import (
    ChessNet,
    ESTNet,
    SEBlock,
    ResidualBlock,
    create_model,
    load_model_with_compatibility,
)

# MCTS
from .mcts import (
    MCTS,
    MCTSNode,
    expand_node,
    select_child,
    simulate,
    run_mcts,
    update_tree,
    select_move_with_mcts,
    generate_mcts_game,
)

# Training utilities
from .training import (
    TRAIN_CONFIG,
    PolicyLoss,
    FocalPolicyLoss,
    ValueLoss,
    EMA,
    create_optimizer,
    create_scheduler,
    train_on_self_play,
)

# Data utilities
from .data import (
    board_to_tensor,
    get_move_index,
    ChessDataset,
    PuzzleDataset,
    SelfPlayDataset,
    load_lichess_puzzles,
    discover_pgn_files,
    load_pgn_games_from_directory,
    load_puzzle_examples_from_directory,
    load_training_examples_from_chess_pgns,
)

# Constants
from .constants import (
    ACTION_SPACE_SIZE,
    MODEL_CONFIG,
    VALID_VARIANTS,
    TRAINING_CONFIG,
    CURRICULUM_CONFIG,
    MCTS_CONFIG,
    SELF_PLAY_CONFIG,
    HARDWARE_CONFIG,
)

# Utilities
from .utils import (
    clear_memory,
    test_tactical_recognition,
    model_summary,
)

__all__ = [
    # Models
    "ChessNet",
    "ESTNet", 
    "SEBlock",
    "ResidualBlock",
    "create_model",
    "load_model_with_compatibility",
    
    # MCTS
    "MCTS",
    "MCTSNode",
    "simulate",
    "run_mcts",
    "update_tree",
    "select_move_with_mcts",
    "generate_mcts_game",
    
    # Training
    "TRAIN_CONFIG",
    "PolicyLoss",
    "FocalPolicyLoss",
    "ValueLoss",
    "EMA",
    "create_optimizer",
    "create_scheduler",
    "train_on_self_play",
    
    # Data
    "board_to_tensor",
    "get_move_index",
    "ChessDataset",
    "PuzzleDataset",
    "SelfPlayDataset",
    "load_lichess_puzzles",
    "discover_pgn_files",
    "load_pgn_games_from_directory",
    "load_puzzle_examples_from_directory",
    "load_training_examples_from_chess_pgns",
    
    # Constants
    "ACTION_SPACE_SIZE",
    "MODEL_CONFIG",
    "VALID_VARIANTS",
    "TRAINING_CONFIG",
    "CURRICULUM_CONFIG",
    "MCTS_CONFIG",
    "SELF_PLAY_CONFIG",
    "HARDWARE_CONFIG",
    
    # Utils
    "clear_memory",
    "test_tactical_recognition",
    "model_summary",
]
