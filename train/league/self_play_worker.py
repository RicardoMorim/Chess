import sys, time, logging, threading
import numpy as np
import torch, chess
from pathlib import Path
from typing import Callable, Dict, Any, Optional, List, Tuple
from multiprocessing import Queue

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import board_to_tensor, get_move_index
from core.constants import ACTION_SPACE_SIZE

logger = logging.getLogger(__name__)


class BatchedGPUEvaluator:
    """Thread-safe evaluator that accumulates boards across MCTS threads.

    Multiple MCTS threads call evaluate() concurrently.  Boards are accumulated
    via the client's batched API and sent to the GPU server in a single
    __BATCH__ message.

    A background collector thread drains completed results and wakes waiting
    callers via threading.Event.
    """

    def __init__(
        self,
        client,
        batch_flush_size: int = 8,
        timeout_sec: float = 30.0,
    ):
        self._client = client
        self._timeout_sec = timeout_sec
        self._lock = threading.Lock()
        self._pending: Dict[str, threading.Event] = {}
        self._results: Dict[str, Tuple[Optional[List[float]], float]] = {}
        self._running = True

        self._collector = threading.Thread(target=self._collect_loop, daemon=True)
        self._collector.start()

    def evaluate(
        self,
        features,
        legal_indices: List[int],
    ) -> Tuple[Optional[List[float]], float]:
        rid = self._client.evaluate_features_batched(features, legal_indices)
        event = threading.Event()
        with self._lock:
            self._pending[rid] = event
        if not event.wait(timeout=self._timeout_sec):
            with self._lock:
                self._pending.pop(rid, None)
            return None, 0.0
        with self._lock:
            logits, value = self._results.pop(rid, (None, 0.0))
            return logits, value

    def _collect_loop(self):
        while self._running:
            time.sleep(0.003)
            with self._lock:
                rids = list(self._pending.keys())
            if not rids:
                continue
            self._client.flush()
            results = self._client.collect_results(rids)
            with self._lock:
                for rid, (logits, value) in results.items():
                    if rid in self._pending:
                        self._results[rid] = (logits, value)
                        self._pending[rid].set()
                        del self._pending[rid]

    def close(self):
        self._running = False


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
    mcts_config = mcts_config or {
        "num_visits": 800,
        "temperature": 1.0,
        "temperature_move_threshold": 30,
        "max_moves": 200,
        "c_puct": 4.0,
        "dirichlet_alpha": 0.3,
        "add_noise": True,
        "parallel_workers": 1,
    }

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
            timeout_warning_budget = {"count": 0}

            batch_flush_size = int(gpu_eval.get("batch_flush_size", 8))
            batched_eval = BatchedGPUEvaluator(
                client,
                batch_flush_size=batch_flush_size,
                timeout_sec=timeout_sec,
            )

            def evaluate_fn(board: chess.Board):
                legal_moves = list(board.legal_moves)
                legal_indices = [get_move_index(m) for m in legal_moves]
                features = board_to_tensor(board, board.fullmove_number, input_channels_remote)
                logits, value = batched_eval.evaluate(features, legal_indices)
                if logits is None:
                    timeout_warning_budget["count"] += 1
                    if timeout_warning_budget["count"] <= 3 or timeout_warning_budget["count"] % 50 == 0:
                        logger.warning(
                            f"Worker {worker_id}: GPU evaluation returned None (timeout or error) "
                            f"[count={timeout_warning_budget['count']}]"
                        )
                return logits, value

            logger.info(f"Worker {worker_id}: Using GPU-batched evaluator (batch={batch_flush_size})")
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
                game_trajectory = play_game_batch_mcts(
                    mcts, device, model_config, worker_id,
                    gpu_eval=gpu_eval, mcts_config=mcts_config,
                )
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


def play_game_batch_mcts(mcts, device, model_config, worker_id, gpu_eval: Optional[Dict[str, Any]] = None,
                          mcts_config: Optional[Dict[str, Any]] = None):
    """
    Play a single self-play game using MCTS.

    CRITICAL: Must use mcts.search() to get MCTS-guided moves, NOT just policy sampling.

    Implements AlphaZero temperature annealing: τ=1 for the first
    `temperature_move_threshold` half-moves (exploration), then τ=0
    (greedy, argmax of visit counts) for the remainder.
    """
    import chess
    mcts_config = mcts_config or {}
    max_moves = int(mcts_config.get("max_moves", 200))
    initial_temperature = float(mcts_config.get("temperature", 1.0))
    temperature_move_threshold = int(mcts_config.get("temperature_move_threshold", 30))

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
        # AlphaZero temperature schedule: τ=initial_temperature for the
        # opening, then τ=0 (greedy) for the rest of the game.
        current_temperature = initial_temperature if move_count < temperature_move_threshold else 0.0

        try:
            # MCTS search - this is the KEY difference
            # Returns policy (4672-dim) and selected move
            policy, selected_move = mcts.search(board, temperature=current_temperature)

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

    # Determine outcome and backfill values.
    # NOTE: A game that hits max_moves MUST be recorded as a draw, regardless
    # of any heuristic material evaluation. Labelling it as a win/loss
    # would corrupt the value targets (z) used for training.
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
            # Game hit max_moves without a real result. Force a draw.
            outcome = "1/2-1/2"
            white_result = 0.0
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
