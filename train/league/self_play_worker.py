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
    gpu_eval: Optional[Dict[str, Any]] = None,
) -> None:

    from core.mcts import MCTS

    # Default configs
    model_config = model_config or {"input_channels": 22, "num_blocks": 15, "channels": 256}
    # NOTE: Self-play runs in multiple processes; keep MCTS inner threading low to avoid oversubscription.
    mcts_config = mcts_config or {"num_visits": 800, "temperature": 1.0, "c_puct": 4.0, "dirichlet_alpha": 0.3, "add_noise": True, "parallel_workers": 1}

    try:
        torch.set_num_threads(1)  # avoid CPU oversubscription

        evaluate_fn = None
        model = None

        if gpu_eval is not None:
            from league.gpu_inference_process import GPUInferenceClient

            client = GPUInferenceClient(
                worker_id=worker_id,
                request_queue=gpu_eval["request_queue"],
                response_queue=gpu_eval["response_queue"],
            )

            timeout_sec = float(gpu_eval.get("timeout_sec", 30.0))
            input_channels_remote = int(gpu_eval.get("input_channels", 22))

            def evaluate_fn(board: chess.Board):
                legal_moves = list(board.legal_moves)
                legal_indices = [get_move_index(m) for m in legal_moves]
                features = board_to_tensor(board, board.fullmove_number, input_channels_remote)
                logits, value = client.evaluate_features(
                    features,
                    legal_indices,
                    timeout_sec=timeout_sec,
                )
                if logits is None:
                    logger.warning(f"Worker {worker_id}: GPU evaluation returned None (timeout or error)")
                return logits, value

            logger.info(f"Worker {worker_id}: Using GPU-batched evaluator (batch={gpu_eval.get('batch_size', 8)})")
        else:
            # Load model locally on CPU (legacy path)
            model = model_constructor(**model_config)
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.eval()
            logger.info(f"Worker {worker_id}: Model loaded on {device}")

        mcts = MCTS(
            model=model,
            device=device,
            num_visits=mcts_config["num_visits"],
            c_puct=mcts_config["c_puct"],
            temperature=mcts_config["temperature"],
            dirichlet_alpha=mcts_config["dirichlet_alpha"],
            add_noise=mcts_config["add_noise"],
            parallel_workers=mcts_config.get("parallel_workers", 1),
            evaluate_fn=evaluate_fn,
        )

        for game_idx in range(num_games):
            try:
                logger.info(f"Worker {worker_id}: Starting game {game_idx+1}/{num_games}")
                game_trajectory = play_game_batch_mcts(mcts, device, model_config, worker_id, gpu_eval=gpu_eval)
                if game_trajectory:
                    result_queue.put({"game_data": game_trajectory, "worker_id": worker_id, "game_idx": game_idx})
                    logger.info(f"Worker {worker_id}: Game {game_idx+1}/{num_games} finished ({game_trajectory['moves']} moves, outcome={game_trajectory['outcome']}, reason={game_trajectory['end_reason']})")
                else:
                    logger.warning(f"Worker {worker_id}: Game {game_idx+1}/{num_games} returned None")
                    result_queue.put({"error": f"Game {game_idx} returned None", "worker_id": worker_id})
            except Exception as game_err:
                logger.error(f"Worker {worker_id}: Game {game_idx+1}/{num_games} failed: {game_err}", exc_info=True)
                result_queue.put({"error": f"Game {game_idx} failed: {str(game_err)}", "worker_id": worker_id})

    except Exception as e:
        logger.error(f"Worker {worker_id} failed: {e}", exc_info=True)
        result_queue.put({"error": str(e), "worker_id": worker_id})


def play_game_batch_mcts(mcts, device, model_config, worker_id, gpu_eval: Optional[Dict[str, Any]] = None):
    """
    Play a single self-play game using MCTS.
    
    CRITICAL: Must use mcts.search() to get MCTS-guided moves, NOT just policy sampling.
    """
    import chess
    max_moves = 120
    game_data = []
    board = chess.Board()
    move_count = 0
    recent_values = []
    resignation_threshold = -0.8
    resignation_count_needed = 2
    resigned = False
    end_reason = "unknown"

    # If using remote evaluator, mcts.model may be None; use explicit input_channels.
    if gpu_eval is not None and "input_channels" in gpu_eval:
        input_channels = int(gpu_eval["input_channels"])
    else:
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
                resigned = True
                end_reason = "resign"
                break
        
        # Make the move
        board.push(selected_move)
        move_count += 1

    def _material_balance(b):
        values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
        }
        white = 0.0
        black = 0.0
        for piece_type, val in values.items():
            white += val * len(b.pieces(piece_type, chess.WHITE))
            black += val * len(b.pieces(piece_type, chess.BLACK))
        return white - black

    # Determine outcome and backfill values
    if resigned:
        # Side to move resigned; opponent wins.
        if board.turn == chess.WHITE:
            outcome = "0-1"
            white_result = -1.0
        else:
            outcome = "1-0"
            white_result = 1.0
    else:
        outcome = board.result()
        if outcome == "1-0":
            white_result = 1.0
            end_reason = "checkmate"
        elif outcome == "0-1":
            white_result = -1.0
            end_reason = "checkmate"
        elif outcome == "1/2-1/2":
            white_result = 0.0
            end_reason = "draw"
        else:
            # Adjudicate if game hit move cap (board.result() == "*")
            score = _material_balance(board)
            score_norm = max(-1.0, min(1.0, score / 39.0))
            if score_norm > 0.2:
                outcome = "1-0"
            elif score_norm < -0.2:
                outcome = "0-1"
            else:
                outcome = "1/2-1/2"
            white_result = score_norm
            end_reason = "max_moves"
    
    # Backfill values (alternate signs for alternating colors)
    trajectory = []
    for idx, (position, policy, _) in enumerate(game_data):
        # Odd indices are black moves, so negate the value
        value = white_result if idx % 2 == 0 else -white_result
        trajectory.append((position, policy, value))
    
    return {
        "trajectory": trajectory,
        "outcome": outcome,
        "end_reason": end_reason,
        "moves": move_count,
    }
