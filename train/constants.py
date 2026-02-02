import chess

# ============================================================================
# CORE CONSTANTS (FROZEN FOR REPRODUCIBILITY)
# ============================================================================
ACTION_SPACE_SIZE = 4672  # Standard move encoding
DIRICHLET_ALPHA = 10 / ACTION_SPACE_SIZE  # ~0.002
DIRICHLET_EPSILON = 0.25

# ============================================================================
# MODEL CONFIGURATION (by variant, not size)
# ============================================================================
MODEL_CONFIG = {
    # Baseline: 18 input channels (standard)
    'baseline': {
        'num_blocks': 15,
        'channels': 256,
        'input_channels': 18,
        'use_se': True,
    },
    # Attack: 22 input channels (with attack maps)
    'attack': {
        'num_blocks': 15,
        'channels': 256,
        'input_channels': 22,
        'use_se': True,
    },
    # EST: Early Split Trunk (experimental)
    'est': {
        'num_blocks': 15,
        'channels': 256,
        'input_channels': 18,
        'use_se': True,
    },
}

# Valid model variants (no legacy aliases)
VALID_VARIANTS = ['baseline', 'attack', 'est']

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    # Optimizer (AlphaZero uses SGD)
    'optimizer': 'sgd',
    'sgd_lr': 0.01,
    'sgd_momentum': 0.9,
    'weight_decay': 1e-4,
    
    # Learning rate schedule
    'lr_schedule': 'cosine',
    'lr_warmup_epochs': 5,
    
    # Gradient clipping
    'grad_clip': 1.0,
    
    # Batch training
    'epochs_per_batch': 5,
    'puzzle_batches_per_game_batch': 5,
}

# ============================================================================
# 3-PHASE CURRICULUM CONFIGURATION
# ============================================================================
CURRICULUM_CONFIG = {
    # Phase 1: Puzzle Bootcamp (isolated)
    'phase1_epochs': 50,
    'phase1_batch_size': 256,
    'phase1_checkmate_bootcamp': True,  # Run intensive checkmate training
    'phase1_target_accuracy': 0.75,     # Target tactical accuracy before Phase 2
    
    # Phase 2: Transition (brief handoff)
    'phase2_epochs': 10,
    'phase2_games': 500,                # Generate initial self-play games
    'phase2_mcts_sims': 400,            # MCTS simulations for transition games
    
    # Phase 3: Pure Self-Play (repeat forever)
    'phase3_games_per_iteration': 200,  # Games per self-play iteration
    'phase3_training_epochs': 5,        # Epochs per iteration
    'phase3_batch_size': 512,           # Large batch for RTX 5080
    'phase3_mcts_sims': 800,            # Full MCTS for quality games
    'phase3_checkmate_interval': 5,     # Checkmate reinforcement every N iterations (0=disabled)
    'phase3_evaluation_interval': 10,   # Evaluate model every N iterations
}

# ============================================================================
# MCTS CONFIGURATION
# ============================================================================
MCTS_CONFIG = {
    # Search parameters
    'num_simulations': 800,       # Simulations per move (training)
    'num_simulations_play': 1600, # Simulations per move (play)
    'c_puct': 2.5,                # Exploration constant
    
    # Parallelization
    'parallel_workers': 16,
    'virtual_loss': 3.0,
    
    # Exploration noise (use frozen DIRICHLET_ALPHA and DIRICHLET_EPSILON)
    'dirichlet_alpha': DIRICHLET_ALPHA,
    'dirichlet_epsilon': DIRICHLET_EPSILON,
    
    # Early stopping
    'early_stop_threshold': 0.9,
    'min_simulations': 100,
    
    # Endgame sim cap
    'legal_moves_sim_cap_threshold': 12,  # if legal_moves < 12: sims //= 2
}

# ============================================================================
# SELF-PLAY CONFIGURATION
# ============================================================================
SELF_PLAY_CONFIG = {
    # Temperature schedule (τ≤0.05 → greedy)
    'temp_threshold_1': 15,        # Moves 1-15: τ=1.0
    'temp_threshold_2': 30,        # Moves 16-30: τ=0.1
    'temp_greedy_threshold': 0.05, # Below this → greedy (one-hot)
    
    # Game limits
    'max_moves': 200,
    'min_game_length': 10,
    
    # Resignation (disabled first 15 moves)
    'resignation_threshold': -0.9,
    'resignation_consecutive': 3,
    'resignation_min_move': 15,
    
    # Data quality filter
    'min_unique_positions': 8,
    'min_root_visits_ratio': 0.7,  # Discard if max_visits < sims * 0.7
    
    # Reward shaping
    'use_reward_shaping': True,
    'discount_factor': 0.99,
}

# ============================================================================
# HARDWARE CONFIGURATION (RTX 5080 16GB + Ultra 9 24-core)
# ============================================================================
HARDWARE_CONFIG = {
    # GPU settings
    'max_batch_size': 512,           # RTX 5080 can handle large batches
    'enable_amp': True,              # Mixed precision for Tensor Cores
    'compile_model': True,           # torch.compile for 2x speedup
    
    # CPU settings
    'dataloader_workers': 8,         # Ultra 9 has 24 cores, use 8 for loading
    'selfplay_workers': 20,          # 20 parallel workers for self-play
    'mcts_parallel_workers': 16,     # Parallel MCTS simulations
    
    # Memory
    'pin_memory': True,
    'prefetch_factor': 4,
}

# ============================================================================
# TACTICAL TEST POSITIONS
# ============================================================================
# Dictionary of tactical test positions with categories
TACTICAL_TEST_POSITIONS = {
    # Checkmate patterns (verified forced mates)
    "mate_in_one": [
        # Back-rank mate (fixed)
        ("3r2k1/5ppp/4p3/8/8/8/5PPP/3R2K1 w - - 0 1", "d1d8"),
        # Anastasia's mate (queen + rook)
        ("2r5/4Nppk/4pn2/8/8/4K3/8/3R1Q2 w - - 0 1", "f1h1"),
        ("2r4k/5Q2/4R3/8/8/4K3/8/8 w - - 0 1", "e6h6"),
        ("7k/6r1/5r2/8/8/Q7/7K/8 b - - 0 1", "f6h6"),
        # Smothered mate (knight checkmate)
        ("5rk1/5pp1/8/5N2/8/8/5PP1/4K2R w - - 0 1", "f5e7"),
        # Classic two-rook checkmate
        ("7k/5Rpp/8/8/8/8/5RPP/7K w - - 0 1", "f7f8"),
        ("6k1/5Rpp/8/8/8/8/5RPP/7K w - - 0 1", "f7f8"),
    ],
    
    # Knight forks (undefended pieces)
    "knight_fork": [
        # Fork king + queen
        ("r3k2r/ppp2ppp/2n5/3N4/4q3/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5c7"),
        # Fork king + rook
        ("r3k2r/pp3ppp/2n1b3/3N4/8/8/PPP2PPP/R3K2R w KQkq - 0 1", "d5c7"),
        # Fork queen + rook (smothered setup)
        ("r1bqkbnr/ppppnppp/4p3/7N/8/1P6/PBP1PPPP/RN1QKB1R b KQkq - 0 1", "h5g7"),
        # Fork two rooks
        ("r3k2r/ppp2ppp/2n5/3N4/8/8/PPP2PPP/2KR3R w kq - 0 1", "d5c7"),
        ("r3k2r/p1p2ppp/8/8/2N1n3/8/PPP2PPP/2KR3R b kq - 0 1", "e4f2"),
    ],
    
    # Absolute pins (pinned to king)
    "pin": [
        ("r3k2r/ppp2ppp/2q1b3/8/8/2N5/PPB2PPP/R3K2R w KQkq - 0 1", "c2a4"),
        ("r2k3r/ppp2ppp/2nqb3/8/8/2N5/PPP2PPP/R3K2R w KQ - 0 1", "a1d1"),
        ("r1bqk2r/ppp1bppp/2n5/3p4/3P4/2N1PN2/PP3PPP/R2QKB1R w KQkq - 0 1", "f1b5"),

    ],
    
    # Discovered attacks/checks (verified)
    "discovered": [
        ("1k5r/1pp2ppp/p7/8/8/2N5/PPPR1PPP/2KR4 b - - 0 1", "d2d8"),
        # Pawn move reveals rook check
        ("r3k2r/ppp1qppp/4n3/8/N7/8/PPP2PPP/R3K2R b KQkq - 0 1", "e6c5"),
    ],
    
    # Skewers (verified)
    "skewer": [
        # rook skewers king + rook
        ("1k1r3r/ppp2ppp/2n5/8/8/2N5/PPPR1PPP/2KR4 b - - 0 1", "d2d8"),
        # Rook skewers king + bishop
        ("r2k3r/ppp2ppp/2nqb3/8/8/2P1N3/PP2KPPP/R6R b - - 0 1", "a1d1"),
    ],
    
    # Endgame tactics (verified)
    "endgame": [
        # Opposition (king vs king)
        ("7k/5R2/6K1/8/8/8/8/8 b - - 0 1", "f7f8"),
    ]
}

# ============================================================================
# MOVE INDEX MAPPING
# ============================================================================
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
