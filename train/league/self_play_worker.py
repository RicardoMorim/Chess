import sys, time, logging, numpy as np
import torch, chess
from pathlib import Path
from typing import Callable, Dict, Any, Optional
from multiprocessing import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import board_to_tensor, get_move_index
from core.constants import ACTION_SPACE_SIZE

logger = logging.getLogger(__name__)

# --------------------------
# GPU batch MCTS self-play
# --------------------------
def self_play_worker(
    model_state_dict: Dict[str, torch.Tensor],
    model_constructor: Callable,
    num_games: int,
    device: str,
    result_queue: Queue,
    model_config: Optional[Dict[str, Any]] = None,
    mcts_config: Optional[Dict[str, Any]] = None,
    worker_id: int = 0,
) -> None:

    from core.mcts import MCTS

    # Default configs
    model_config = model_config or {"input_channels": 22, "num_blocks": 15, "channels": 256}
    mcts_config = mcts_config or {"num_visits": 800, "temperature": 1.0, "c_puct": 4.0, "dirichlet_alpha": 0.3, "add_noise": True, "parallel_workers": 8}

    try:
        # Load model on GPU (or CPU)
        torch.set_num_threads(1)  # avoid CPU oversubscription
        model = model_constructor(**model_config)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        # NOTE: Keep in FP32 for stability. FP16 only helpful with GPU batching of MCTS.

        mcts = MCTS(
            model=model,
            device=device,
            num_visits=mcts_config["num_visits"],
            c_puct=mcts_config["c_puct"],
            temperature=mcts_config["temperature"],
            dirichlet_alpha=mcts_config["dirichlet_alpha"],
            add_noise=mcts_config["add_noise"],
            parallel_workers=mcts_config.get("parallel_workers", 8),
        )

        logger.info(f"Worker {worker_id}: Model loaded on {device}")

        for game_idx in range(num_games):
            game_trajectory = play_game_batch_mcts(mcts, device, model_config, worker_id)
            if game_trajectory:
                result_queue.put({"game_data": game_trajectory, "worker_id": worker_id, "game_idx": game_idx})
                logger.info(f"Worker {worker_id}: Game {game_idx+1}/{num_games} finished ({len(game_trajectory)} moves)")

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        result_queue.put({"error": str(e), "worker_id": worker_id})


def play_game_batch_mcts(mcts, device, model_config, worker_id):
    """
    Play a single self-play game using MCTS.
    
    CRITICAL: Must use mcts.search() to get MCTS-guided moves, NOT just policy sampling.
    """
    import chess
    max_moves = 150
    game_data = []
    board = chess.Board()
    move_count = 0
    recent_values = []
    resignation_threshold = -0.9
    resignation_count_needed = 3

    input_channels = getattr(mcts.model, "input_channels", model_config.get("input_channels", 22))

    while not board.is_game_over() and move_count < max_moves:
        try:
            # MCTS search - this is the KEY difference
            # Returns policy (4672-dim) and selected move
            policy, selected_move = mcts.search(board)
            
        except Exception as e:
            logger.error(f"Worker {worker_id}: MCTS search failed at move {move_count}: {e}")
            logger.error("Aborting game")
            return None
        
        if selected_move is None or policy is None:
            logger.error(f"Worker {worker_id}: MCTS returned None at move {move_count}")
            return None
        
        # Store board position and MCTS policy
        position = board_to_tensor(board, board.fullmove_number, input_channels)
        game_data.append((position, policy, None))  # Value filled in later
        
        # Resignation logic using MCTS value estimate
        if hasattr(mcts, '_last_value'):
            recent_values.append(mcts._last_value)
            recent_values = recent_values[-resignation_count_needed:]
            if len(recent_values) >= resignation_count_needed and all(v < resignation_threshold for v in recent_values):
                logger.info(f"Worker {worker_id}: Resignation at move {move_count} (value={recent_values[-1]:.3f})")
                break
        
        # Make the move
        board.push(selected_move)
        move_count += 1

    # Determine outcome and backfill values
    outcome = board.result()
    if outcome == "1-0":
        white_result = 1.0
    elif outcome == "0-1":
        white_result = -1.0
    else:
        white_result = 0.0
    
    # Backfill values (alternate signs for alternating colors)
    trajectory = []
    for idx, (position, policy, _) in enumerate(game_data):
        # Odd indices are black moves, so negate the value
        value = white_result if idx % 2 == 0 else -white_result
        trajectory.append((position, policy, value))
    
    return trajectory
