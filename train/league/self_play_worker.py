"""
Parallel Self-Play Worker
==========================

This module runs on CPU cores and generates game data using MCTS.
Workers are completely stateless except for one batch of the model.

Design principles:
- No optimizer or training (CPU only)
- No replay buffer access (data queued out)
- Minimal memory footprint
- Truly parallelizable across N workers
"""

import sys
import torch
import chess
import logging
from pathlib import Path
from typing import Callable, Dict, Any, Optional

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import board_to_tensor, get_move_index
from core.constants import ACTION_SPACE_SIZE


logger = logging.getLogger(__name__)


def self_play_worker(
    model_state_dict: Dict[str, torch.Tensor],
    model_constructor: Callable,
    num_games: int,
    device: str,
    result_queue: Any,
    model_config: Optional[Dict[str, Any]] = None,
    mcts_config: Optional[Dict[str, Any]] = None,
    worker_id: int = 0,
) -> None:
    """
    Run self-play games and queue results to parent process.
    
    This function runs on a separate CPU process. It:
    1. Loads a model snapshot
    2. Plays num_games of chess using MCTS
    3. Queues each game's trajectory to result_queue
    4. Never touches optimizer or replay buffer
    
    Args:
        model_state_dict: State dict of the frozen model
        model_constructor: Function that returns a new model instance
        num_games: Number of games to play
        device: Device to run on ("cpu" or "cuda")
        result_queue: multiprocessing.Queue to send game data
        model_config: Dict with model hyperparameters (channels, blocks, etc.)
        mcts_config: Dict with MCTS hyperparameters (visits, temperature, etc.)
        worker_id: Identifier for logging
    
    Returns:
        None (results sent via queue)
    """
    
    # Import here to avoid circular dependencies
    from core.mcts import MCTS
    
    # Default configs
    if model_config is None:
        model_config = {
            "input_channels": 22,
            "num_blocks": 15,
            "channels": 256,
        }
    
    if mcts_config is None:
        mcts_config = {
            "num_visits": 800,
            "temperature": 1.0,
            "c_puct": 4.0,
            "dirichlet_alpha": 0.3,
            "add_noise": True,
        }
    
    try:
        # Load model on this process
        model = model_constructor(**model_config)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        
        logger.info(f"Worker {worker_id}: Model loaded, device={device}")
        
        # Create MCTS searcher
        mcts = MCTS(
            model=model,
            device=device,
            num_visits=mcts_config["num_visits"],
            c_puct=mcts_config["c_puct"],
            temperature=mcts_config["temperature"],
            dirichlet_alpha=mcts_config["dirichlet_alpha"],
            add_noise=mcts_config["add_noise"],
        )
        
        logger.info(f"Worker {worker_id}: Starting {num_games} games")
        
        # Play games
        for game_idx in range(num_games):
            game_trajectory = _play_single_game(mcts, model, device, model_config, result_queue, worker_id)
            
            if game_trajectory is not None:
                # Queue the game data to parent process
                result_queue.put({
                    "game_data": game_trajectory,
                    "worker_id": worker_id,
                    "game_idx": game_idx,
                })
                
                logger.info(f"Worker {worker_id}: Game {game_idx+1}/{num_games} complete ({len(game_trajectory)} moves)")
            else:
                logger.error(f"Worker {worker_id}: Game {game_idx+1} returned None trajectory")
                return
        
        logger.info(f"Worker {worker_id}: ✓ Completed all {num_games} games")
    
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
        result_queue.put({"error": str(e), "worker_id": worker_id})


def _play_single_game(
    mcts: Any,
    model: Any,
    device: str,
    model_config: Dict[str, Any],
    result_queue: Any = None,
    worker_id: int = 0,
) -> list:
    """
    Play a single self-play game and return trajectory.
    
    Hard limits:
    - Max 150 moves (normal chess games are 40-60 moves)
    - Max 5 minutes wall clock
    - Early resignation if mate threat detected
    
    Args:
        mcts: MCTS searcher instance
        model: Neural network model
        device: Compute device
        model_config: Model configuration dict
    
    Returns:
        List of (position, policy, value) tuples for the game
    """
    import time
    
    game_data = []
    board = chess.Board()
    move_count = 0
    max_moves = 150  # Hard limit (normal games are 40-60)
    game_start_time = time.time()
    max_game_time = 300  # 5 minutes max per game
    
    input_channels = getattr(model, "input_channels", model_config.get("input_channels", 22))
    
    while not board.is_game_over() and move_count < max_moves:
        # Check time limit
        if time.time() - game_start_time > max_game_time:
            logger.warning(f"Worker {worker_id}: Game timeout after {move_count} moves")
            break
        
        # Get current board state
        position = board_to_tensor(board, move_number=board.fullmove_number, input_channels=input_channels)
        
        # Run MCTS to get move probabilities
        try:
            policy, move = mcts.search(board)
        except Exception as e:
            logger.error(f"MCTS FAILURE at move {move_count}: {e}")
            logger.error("Aborting worker -- training signal compromised")
            if result_queue is not None:
                result_queue.put({
                    "error": f"MCTS failure: {str(e)}",
                    "worker_id": worker_id,
                })
            return None
        
        if move is None or policy is None:
            logger.error(f"MCTS returned None at move {move_count}")
            logger.error("Aborting worker")
            if result_queue is not None:
                result_queue.put({
                    "error": "MCTS returned None",
                    "worker_id": worker_id,
                })
            return None
        
        # Store position and policy
        game_data.append((position, policy, None))  # Value set later
        
        # Make the move
        board.push(move)
        move_count += 1
    
    # Determine game outcome and backfill values
    outcome = board.result()
    if outcome == "1-0":
        white_result = 1.0
    elif outcome == "0-1":
        white_result = -1.0
    else:
        white_result = 0.0
    
    # Backfill values (alternate signs for alternating colors)
    game_trajectory = []
    for idx, (position, policy, _) in enumerate(game_data):
        # Odd indices are black moves, so negate the value
        value = white_result if idx % 2 == 0 else -white_result
        game_trajectory.append((position, policy, value))
    
    return game_trajectory


def _create_uniform_policy(board: chess.Board) -> Any:
    """
    Create a uniform policy over legal moves.
    
    Args:
        board: Chess board
    
    Returns:
        numpy array of shape (4672,) with uniform probabilities for legal moves
    """
    import numpy as np
    
    policy = np.zeros(4672, dtype=np.float32)
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return policy
    
    uniform_prob = 1.0 / len(legal_moves)
    
    for move in legal_moves:
        move_idx = get_move_index(move)
        if move_idx < 4672:
            policy[move_idx] = uniform_prob
    
    return policy
