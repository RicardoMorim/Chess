"""
League Trainer - Main Training Loop
====================================

Orchestrates parallel self-play, replay buffer management, model training,
checkpointing, and low-frequency evaluation.

THE RULE ENFORCED HERE:
Only self-play with MCTS improves models long-term.
Everything else (bootstrap, evaluation) only reduces cold start, catches regressions,
or measures strength.

Architecture:
- Self-play workers: CPU parallel, MCTS-based
- Training loop: GPU sequential, batched from replay buffer
- Evaluation: Low-frequency, frozen checkpoints only
- Checkpointing: Periodic snapshots for replay and regression detection
"""

import math
import torch
import torch.multiprocessing as mp
import torch.optim as optim
import chess
from torch.optim.lr_scheduler import LambdaLR
from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from league.replay_buffer import ReplayBuffer
from league.self_play_worker import self_play_worker
from league.monitoring import MetricsCollector
from league.evaluator import Evaluator
from league.evolution_logger import EvolutionLogger
from league.gpu_inference_process import gpu_inference_server_main
from league.datasets import AuxDataConfig, AuxDataLoader

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


class LeagueTrainer:
    """
    Main training orchestrator for parallel league training.
    
    Manages:
    - N self-play workers (CPU-bound)
    - M training loops (GPU-bound, one per model variant)
    - Periodic checkpointing
    - Evaluation and regression detection
    - Metrics collection and monitoring
    """
    
    # Training hyperparameters (AlphaZero-faithful, scaled for 1 GPU)
    VARIANTS = ["baseline", "attack", "est"]
    # Parallelism config (INTERMEDIATE - validated and stable)
    # NOTE: Self-play already multiplies across variants; keep this modest to avoid oversubscription.
    NUM_SELF_PLAY_WORKERS = 6  # CPU processes per variant for self-play
    GAMES_PER_WORKER_PER_ROUND = 5  # Multiple games per worker
    BATCH_SIZE = 256  # AlphaZero uses 4096; we use 256 (16× smaller, RTX 5080 sweet spot)
    TRAINING_STEPS_PER_ROUND = 50  # More updates per round to use larger LR effectively
    CHECKPOINT_EVERY_N_ROUNDS = 5  # Checkpoint every 5 rounds
    EVAL_EVERY_N_ROUNDS = 100  # Skip for now
    METRICS_EVERY_N_ROUNDS = 5
    BUFFER_SAVE_EVERY_N_ROUNDS = 5

    # Loss weighting
    # Policy head has 4672 outputs, value head has 1 — equal weighting
    # means value gradient is ~4672x smaller. These multipliers compensate.
    POLICY_LOSS_WEIGHT = 1.0
    VALUE_LOSS_WEIGHT = 10.0  # Boost value head gradient

    # LR schedule (AlphaZero paper: 0.2 -> 0.02 -> 0.002 -> 0.0002 at steps 100k/300k/500k).
    # We scale this 8× for our regime (one GPU vs 64 TPUs): initial 0.025, drops 5× at
    # step 1000 and again at step 3000. Total budget ~5000 steps.
    INITIAL_LR = 0.025
    LR_DROP_FACTOR = 0.2  # multiply LR by this at each milestone
    LR_MILESTONE_1 = 1000  # step at which first LR drop happens
    LR_MILESTONE_2 = 3000  # step at which second LR drop happens
    LR_FINAL = 0.001  # floor LR after all drops
    LR_WARMUP_STEPS = 100  # Linear warmup before decay (AlphaZero paper implicitly warms up via the 100k milestone)

    # Devices / concurrency
    SELF_PLAY_DEVICE = "cpu"  # Safer: avoids many CUDA contexts across worker processes
    SELF_PLAY_VARIANT_PARALLELISM = 3  # How many variants generate self-play concurrently

    # When GPU batching is enabled, self-play is often bottlenecked by request concurrency.
    # More CPU workers helps fill GPU batches even if CPU utilization stays moderate.
    GPU_SELF_PLAY_WORKERS = 14

    # MCTS hyperparameters (AlphaZero paper: 800 sims/move for both training and eval).
    # We use 200 for self-play (4× less than paper, scaled for 1 GPU) and 400 for eval
    # (half of paper — enough for fair strength signal without dominating wall time).
    MCTS_VISITS_SELFPLAY = 200  # Mid-point between 80 (too few) and 800 (paper) — better checkmate signal
    MCTS_VISITS_EVAL = 400  # Half of paper's 800; ample for tournament-grade eval
    C_PUCT = 4.0
    # Temperature schedule (AlphaZero paper: τ=1 for first 30 half-moves, then 0=greedy).
    # Self-play workers enforce this schedule per-move.
    TEMPERATURE_INITIAL = 1.0
    TEMPERATURE_MOVE_THRESHOLD = 30  # move index after which we drop to greedy
    DIRICHLET_ALPHA = 0.3

    # Game length cap (AlphaZero paper: 512 for chess; 200 is a practical compromise that
    # still allows most natural games to finish via checkmate or resignation).
    MAX_GAME_MOVES = 200

    # Replay buffer config (AlphaZero paper: 500k games ≈ 44M positions).
    # We use 100k positions per variant (~3000 games of 30 plies each). 1× GPU can't
    # process 44M positions in a reasonable time, but 100k is enough to retain diversity
    # across rounds.
    REPLAY_BUFFER_MAX_SIZE = 100_000

    # Checkpoint retention
    CHECKPOINT_KEEP_LAST_N = 3
    CHECKPOINT_KEEP_EVERY_N = 15
    CHECKPOINT_ALWAYS_KEEP_STEPS = {1}
    
    # Disk usage safeguards
    MAX_BUFFER_FILES_PER_VARIANT = 3  # Keep only the 3 most recent buffer files
    DISK_USAGE_CHECK_EVERY_N_ROUNDS = 10  # Check disk space periodically
    CRITICAL_DISK_THRESHOLD_PCT = 5  # Alert if free disk < this (%)
    
    # Adaptive MCTS visitation tuning
    TARGET_GAMES_PER_MINUTE = 10  # Target throughput (games/min)
    ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS = 5  # Check and adjust visits periodically
    VISITS_ADJUSTMENT_FACTOR = 0.15  # Adjust visits by 15% each step
    MIN_MCTS_VISITS = 6  # Never go below this
    MAX_MCTS_VISITS = 32  # Never go above this

    # GPU-batched inference (self-play) tuning
    GPU_INFER_BATCH_SIZE = 64
    GPU_INFER_POST_WAIT_MS = 15
    GPU_MCTS_PARALLEL_WORKERS = 2
    GPU_EVAL_TIMEOUT_SEC = 90.0

    # Result collection and stall monitoring
    RESULT_QUEUE_POLL_SEC = 10.0
    RESULT_STALL_WARN_SEC = 180.0

    # ------------------------------------------------------------------
    # Auxiliary data injection (puzzles, pro PGNs, Stockfish eval)
    # Each source has its own on/off toggle. Disable any source by
    # setting the matching *_BATCHES_PER_GAME_BATCH constant to 0 OR
    # flipping the corresponding USE_* flag.
    # ------------------------------------------------------------------
    USE_PUZZLE_INJECTION = True
    USE_PRO_GAMES = True
    USE_STOCKFISH_EVAL = True  # only used for pro-PGN labelling and benchmarking
    PUZZLE_BATCHES_PER_GAME_BATCH = 1  # puzzle batches interleaved per self-play step
    PROGAME_BATCHES_PER_GAME_BATCH = 1  # progame batches interleaved per self-play step
    STOCKFISH_DEPTH_LABEL = 12   # depth when labelling pro PGN positions
    STOCKFISH_DEPTH_BENCH = 15   # depth when benchmarking vs Stockfish
    STOCKFISH_BENCH_NUM_GAMES = 20
    STOCKFISH_BENCH_EVERY_N_ROUNDS = 25
    STOCKFISH_BENCH_TIME_LIMIT_MS = 200
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = "cuda",
        use_gpu_batching: bool = False,
    ):
        """
        Initialize league trainer.
        
        Args:
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for logs and metrics
            device: Device for GPU training ("cuda" or "cpu")
            use_gpu_batching: If True, enable GPU-batched inference server (EXPERIMENTAL)
        """
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.use_gpu_batching = use_gpu_batching
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models, optimizers, schedulers, buffers
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.buffers = {}
        self.model_configs = {}
        
        # GPU inference server (lazy init on first use if enabled)
        self._gpu_inference_server = None
        
        # Metrics and evaluation
        self.metrics = MetricsCollector(str(self.log_dir))
        self.evaluator = Evaluator(device=str(self.device), mcts_visits=self.MCTS_VISITS_EVAL)
        self.evolution_logger = EvolutionLogger(str(self.log_dir))
        
        # Optional W&B tracking via environment configuration
        self.metrics.enable_wandb(
            project=os.environ.get("WANDB_PROJECT", "chess-league"),
            run_name=os.environ.get("WANDB_RUN_NAME"),
            config={
                "device": device,
                "checkpoint_dir": str(self.checkpoint_dir),
                "log_dir": str(self.log_dir),
                "use_gpu_batching": self.use_gpu_batching,
                "variants": self.VARIANTS,
            },
            tags=["self-play", "league", "chess"],
            mode=os.environ.get("WANDB_MODE", "offline"),
        )
        
        # State tracking
        self.round = 0
        self.start_round = 0
        self.total_games = 0
        self.total_training_steps = 0

        # Adaptive self-play controls (may be throttled on high RAM usage)
        self._num_self_play_workers = self.NUM_SELF_PLAY_WORKERS
        self._variant_parallelism = self.SELF_PLAY_VARIANT_PARALLELISM
        self._buffer_target_size = self.REPLAY_BUFFER_MAX_SIZE
        self._last_buffer_target_size = self.REPLAY_BUFFER_MAX_SIZE
        self._last_disk_check = 0  # Track when disk was last checked
        
        # Adaptive MCTS visitation tracking
        self._current_mcts_visits = self.MCTS_VISITS_SELFPLAY
        self._throughput_history = {variant: [] for variant in self.VARIANTS}  # Rolling window of games/min

        # Auxiliary data (puzzles, pro PGNs) — only built if a toggle is on
        self.aux_config = AuxDataConfig(
            use_puzzle_injection=self.USE_PUZZLE_INJECTION,
            use_pro_games=self.USE_PRO_GAMES,
            use_stockfish_eval=self.USE_STOCKFISH_EVAL,
            stockfish_depth=self.STOCKFISH_DEPTH_LABEL,
        )
        self.aux_loader = AuxDataLoader(self.aux_config)
        if self.USE_PUZZLE_INJECTION or self.USE_PRO_GAMES:
            try:
                self.aux_loader.initialize()
                if self.aux_loader.is_ready():
                    logger.info(
                        f"Aux data: puzzles={'on' if self.aux_loader._puzzle_ready() else 'off'} "
                        f"({len(self.aux_loader.puzzle_dataset) if self.aux_loader.puzzle_dataset is not None else 0}), "
                        f"progames={'on' if self.aux_loader._progame_ready() else 'off'} "
                        f"({len(self.aux_loader.progame_dataset) if self.aux_loader.progame_dataset is not None else 0})"
                    )
                else:
                    logger.warning("Aux data toggles enabled but no datasets loaded; injection will be skipped")
            except Exception as e:
                logger.error(f"Aux data initialization failed: {e}", exc_info=True)
                self.aux_loader = AuxDataLoader(self.aux_config)  # fall back to a clean no-op loader

        # Stockfish benchmark helper (lazy; subprocess opened on first use)
        self._stockfish_benchmark = None
        
        if self.use_gpu_batching:
            logger.info("GPU-batched inference enabled (EXPERIMENTAL)")
            # Each variant gets its own dedicated GPU process; they can run in parallel.
            # Keep variant parallelism = 3 (all variants simultaneously)
            self._num_self_play_workers = self.GPU_SELF_PLAY_WORKERS
        else:
            logger.info("Using CPU-only MCTS for self-play (GPU-batched disabled)")
        
        logger.info(f"LeagueTrainer initialized: device={self.device}, checkpoints={self.checkpoint_dir}")

    def _parse_checkpoint_step(self, path: Path, variant: str) -> Optional[int]:
        """Parse step number from '<variant>_step_<step>.pt'. Returns None if it doesn't match."""
        name = path.name
        prefix = f"{variant}_step_"
        if not name.startswith(prefix) or not name.endswith(".pt"):
            return None
        step_str = name[len(prefix):-3]
        try:
            return int(step_str)
        except ValueError:
            return None

    def _prune_checkpoints(self, variant: str) -> None:
        """Prune old checkpoints for a variant using retention policy."""
        try:
            ckpts = list(self.checkpoint_dir.glob(f"{variant}_step_*.pt"))
            if not ckpts:
                return

            steps_and_paths = []
            for p in ckpts:
                step = self._parse_checkpoint_step(p, variant)
                if step is not None:
                    steps_and_paths.append((step, p))

            if not steps_and_paths:
                return

            steps_and_paths.sort(key=lambda x: x[0])
            steps = [s for s, _ in steps_and_paths]
            last_n = set(steps[-self.CHECKPOINT_KEEP_LAST_N:]) if self.CHECKPOINT_KEEP_LAST_N > 0 else set()
            every_n = set(s for s in steps if (self.CHECKPOINT_KEEP_EVERY_N > 0 and s % self.CHECKPOINT_KEEP_EVERY_N == 0))
            keep_steps = set(self.CHECKPOINT_ALWAYS_KEEP_STEPS) | last_n | every_n

            for step, path in steps_and_paths:
                if step in keep_steps:
                    continue
                try:
                    path.unlink(missing_ok=True)
                    logger.info(f"Pruned checkpoint: {path.name}")
                except Exception as e:
                    logger.warning(f"Failed to prune checkpoint {path}: {e}")

                # Prune matching replay buffer file
                buffer_path = self.checkpoint_dir / f"{variant}_buffer_step_{step}.npz"
                try:
                    if buffer_path.exists():
                        buffer_path.unlink(missing_ok=True)
                        logger.info(f"Pruned buffer: {buffer_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to prune buffer {buffer_path}: {e}")
        except Exception as e:
            logger.warning(f"Checkpoint pruning failed for {variant}: {e}")
    
    def initialize_models(
        self,
        model_constructor,
        model_configs: Dict[str, Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize models for each variant.
        
        Args:
            model_constructor: Function that creates a model given **config
            model_configs: Dict mapping variant -> config dict
        """
        if model_configs is None:
            # Default configs for each variant
            model_configs = {
                "baseline": {},
                "attack": {},
                "est": {
                    "shared_blocks": 5,
                    "policy_blocks": 5,
                    "value_blocks": 5,
                },
            }
        
        self._model_constructor = model_constructor

        for variant in self.VARIANTS:
            config = model_configs.get(variant, {})
            config = {"variant": variant, **config}
            
            # Create model (with value dropout for regularization)
            model = model_constructor(**config, value_dropout=0.2)
            model.to(self.device)
            
            # Create optimizer (AlphaZero paper: SGD, momentum=0.9, weight_decay=1e-4)
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.INITIAL_LR,
                momentum=0.9,
                weight_decay=1e-4,
            )

            # LR schedule (AlphaZero paper: linear warmup, then 0.2 -> 0.02 -> 0.002 -> 0.0002
            # at steps 100k/300k/500k of a 700k-step run). We scale the milestones 100× to match
            # our 5k-step budget and the LR_DROP_FACTOR of 0.2 for the same 5× ratio per drop.
            initial_lr = self.INITIAL_LR
            warmup = self.LR_WARMUP_STEPS
            m1 = self.LR_MILESTONE_1
            m2 = self.LR_MILESTONE_2
            drop = self.LR_DROP_FACTOR
            floor = self.LR_FINAL

            def lr_lambda_floored(step):
                if step < warmup:
                    # Linear warmup: LR ramps from 0 to initial_lr
                    return step / warmup
                # Piecewise constant after warmup; each milestone multiplies by drop
                factor = 1.0
                if step >= m1:
                    factor *= drop
                if step >= m2:
                    factor *= drop
                raw_lr = initial_lr * factor
                return max(floor, raw_lr) / initial_lr

            scheduler = LambdaLR(optimizer, lr_lambda_floored)
            
            # Create replay buffer
            buffer = ReplayBuffer(max_size=self.REPLAY_BUFFER_MAX_SIZE)
            
            # Store
            self.models[variant] = model
            self.optimizers[variant] = optimizer
            self.schedulers[variant] = scheduler
            self.buffers[variant] = buffer
            self.model_configs[variant] = config
            
            logger.info(f"Initialized model: {variant}")

        # Create an initial snapshot at step=1 so retention rules (keep step 1) always apply.
        # Do NOT overwrite if it already exists (important for resume).
        for variant in self.VARIANTS:
            try:
                path = self.checkpoint_dir / f"{variant}_step_1.pt"
                if not path.exists():
                    self.save_checkpoint(variant, step=1)
            except Exception as e:
                logger.warning(f"Failed to save initial checkpoint for {variant}: {e}")

    def load_latest_checkpoints(self) -> int:
        """Load latest available checkpoints for each variant (if any).

        Returns:
            The maximum loaded step across variants (0 if none).

        Notes:
            - Replay buffers are not persisted, so they start empty on resume.
            - If only step=1 exists (initial snapshot), training starts from round 0.
        """
        max_step_loaded = 0

        for variant in self.VARIANTS:
            ckpts = list(self.checkpoint_dir.glob(f"{variant}_step_*.pt"))
            steps = []
            for p in ckpts:
                step = self._parse_checkpoint_step(p, variant)
                if step is not None:
                    steps.append((step, p))
            if not steps:
                continue

            steps.sort(key=lambda x: x[0])
            step, path = steps[-1]

            try:
                checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
                    opt_state = checkpoint.get("optimizer_state_dict")
                    sched_state = checkpoint.get("scheduler_state_dict")
                    self.total_games = int(checkpoint.get("total_games", self.total_games))
                    self.total_training_steps = int(checkpoint.get("total_training_steps", self.total_training_steps))
                else:
                    state_dict = checkpoint
                    opt_state = None
                    sched_state = None

                # Load model weights (non-strict for compatibility)
                self.models[variant].load_state_dict(state_dict, strict=False)

                # Load optimizer state if present
                if opt_state is not None and variant in self.optimizers:
                    try:
                        self.optimizers[variant].load_state_dict(opt_state)
                    except Exception as e:
                        logger.warning(f"{variant}: could not load optimizer state: {e}")

                # Load scheduler state if present
                if sched_state is not None and variant in self.schedulers:
                    try:
                        self.schedulers[variant].load_state_dict(sched_state)
                    except Exception as e:
                        logger.warning(f"{variant}: could not load scheduler state: {e}")

                logger.info(f"Resumed {variant} from checkpoint: {path.name}")
                max_step_loaded = max(max_step_loaded, int(step))

                # Try to load replay buffer for the same step
                buffer_path = self.checkpoint_dir / f"{variant}_buffer_step_{step}.npz"
                if buffer_path.exists():
                    try:
                        self.buffers[variant].load_from_npz(str(buffer_path))
                        logger.info(f"Loaded replay buffer: {buffer_path.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load buffer {buffer_path}: {e}")

            except Exception as e:
                logger.warning(f"Failed to load checkpoint {path} for {variant}: {e}")

        # If we only have the initial step=1 snapshot, start at round 0.
        if max_step_loaded <= 1:
            self.start_round = 0
        else:
            self.start_round = max_step_loaded
        self.round = self.start_round

        return max_step_loaded
    
    def generate_self_play(self, variant: str) -> int:
        """
        Launch parallel self-play workers and collect games.
        
        Uses multiprocessing to parallelize across CPU cores.
        
        Args:
            variant: Model variant to generate games for
        
        Returns:
            Number of games generated
        """
        
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        processes = []

        # If GPU batching is enabled, start one GPU inference process for this variant.
        use_gpu_eval = bool(self.use_gpu_batching and self.device.type == "cuda" and torch.cuda.is_available())
        gpu_server_proc = None
        gpu_request_queue = None
        gpu_response_queues = None

        # Self-play feature channels must match the model variant.
        # baseline/est => 18, attack => 22
        input_channels = 22 if variant == "attack" else 18
        
        model = self.models[variant]
        # Always snapshot weights on CPU for multiprocessing.
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if use_gpu_eval:
            gpu_request_queue = ctx.Queue(maxsize=5000)
            gpu_response_queues = [ctx.Queue(maxsize=5000) for _ in range(self._num_self_play_workers)]

            gpu_server_proc = ctx.Process(
                target=gpu_inference_server_main,
                args=(
                    model_state,
                    self._model_constructor,
                    self.model_configs[variant],
                    str(self.device),
                    gpu_request_queue,
                    gpu_response_queues,
                    self.GPU_INFER_BATCH_SIZE,
                    self.GPU_INFER_POST_WAIT_MS,
                    variant,
                    str(self.log_dir / f"gpu_stats_{variant}.json"),
                ),
                name=f"{variant}_gpu_inference",
            )
            gpu_server_proc.start()
            logger.info(f"{variant}: GPU inference process started")
        elif self.use_gpu_batching and self.device.type != "cuda":
            logger.warning(f"{variant}: GPU batching requested but trainer device={self.device}; falling back to CPU-only self-play")
        
        # Launch workers
        for worker_id in range(self._num_self_play_workers):
            gpu_eval_payload = None
            if use_gpu_eval:
                gpu_eval_payload = {
                    "request_queue": gpu_request_queue,
                    "response_queue": gpu_response_queues[worker_id],
                    "timeout_sec": self.GPU_EVAL_TIMEOUT_SEC,
                    "input_channels": input_channels,
                }

            p = ctx.Process(
                target=self_play_worker,
                args=(
                    ({} if use_gpu_eval else model_state),
                    self._model_constructor,
                    self.GAMES_PER_WORKER_PER_ROUND,
                    self.SELF_PLAY_DEVICE,
                    queue,
                    self.model_configs[variant],
                    {
                        "num_visits": self._current_mcts_visits,  # Adaptive MCTS visits
                        "temperature": self.TEMPERATURE_INITIAL,  # initial τ; worker anneals to 0 after TEMPERATURE_MOVE_THRESHOLD
                        "temperature_move_threshold": self.TEMPERATURE_MOVE_THRESHOLD,
                        "c_puct": self.C_PUCT,
                        "dirichlet_alpha": self.DIRICHLET_ALPHA,
                        "add_noise": True,
                        "parallel_workers": (self.GPU_MCTS_PARALLEL_WORKERS if use_gpu_eval else 2),
                    },
                    worker_id,
                    gpu_eval_payload,
                ),
                name=f"{variant}_worker_{worker_id}",
            )
            p.start()
            processes.append(p)
        
        # Collect results
        num_games_expected = self._num_self_play_workers * self.GAMES_PER_WORKER_PER_ROUND
        games_collected = 0
        errors = 0
        game_lengths = []
        sp_start = time.time()
        
        logger.info(f"{variant}: Waiting for {num_games_expected} games from {self._num_self_play_workers} workers...")
        processed_reports = 0
        last_progress_ts = time.time()
        
        while processed_reports < num_games_expected:
            try:
                result = queue.get(timeout=self.RESULT_QUEUE_POLL_SEC)
            except Exception:
                alive_count = sum(1 for p in processes if p.is_alive())

                now = time.time()
                if now - last_progress_ts >= self.RESULT_STALL_WARN_SEC:
                    gpu_status = "N/A"
                    if gpu_server_proc is not None:
                        gpu_status = "ALIVE" if gpu_server_proc.is_alive() else "DEAD"
                    logger.warning(
                        f"{variant}: No new game report for {self.RESULT_STALL_WARN_SEC:.0f}s. "
                        f"Progress {processed_reports}/{num_games_expected} (games={games_collected}, errors={errors}), "
                        f"workers_alive={alive_count}/{len(processes)}, gpu_proc={gpu_status}"
                    )
                    last_progress_ts = now

                if alive_count == 0:
                    logger.error(f"{variant}: All workers stopped before collecting all reports; aborting self-play phase")
                    break
                continue

            processed_reports += 1
            last_progress_ts = time.time()
            
            if "error" in result:
                logger.error(f"Worker {result.get('worker_id', '?')} error: {result['error']}")
                errors += 1
                continue
            
            game_payload = result["game_data"]
            if isinstance(game_payload, dict):
                game_data = game_payload.get("trajectory", [])
                outcome = game_payload.get("outcome", "1/2-1/2")
            else:
                game_data = game_payload
                outcome = "1/2-1/2"

            if not game_data:
                continue

            self.buffers[variant].add_game(game_data)
            
            games_collected += 1
            self.total_games += 1
            game_lengths.append(len(game_data))
            
            # Record metrics
            game_length = len(game_data)
            self.metrics.record_self_play_game(variant, game_length, outcome)
            
            # Progress indicator
            if games_collected % 5 == 0 or games_collected == num_games_expected:
                logger.info(f"{variant}: Collected {games_collected}/{num_games_expected} games...")
        
        # Wait for workers to finish
        for p in processes:
            p.join()
        
        # Free queue memory properly
        try:
            queue.close()
            queue.join_thread()
        except Exception as e:
            logger.warning(f"{variant}: Failed to close queue: {e}")

        # Stop GPU inference process (if any)
        if gpu_request_queue is not None:
            try:
                gpu_request_queue.put(None)
                gpu_request_queue.close()
                gpu_request_queue.join_thread()
            except Exception:
                pass
                
        if gpu_response_queues is not None:
            for q in gpu_response_queues:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        if gpu_server_proc is not None:
            gpu_server_proc.join(timeout=5.0)
            if gpu_server_proc.is_alive():
                logger.warning(f"{variant}: GPU inference process did not stop in time; terminating")
                try:
                    gpu_server_proc.terminate()
                except Exception:
                    pass
        
        if errors > 0:
            logger.warning(f"{variant}: {errors}/{num_games_expected} games failed")
        
        avg_moves = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        elapsed = max(0.001, time.time() - sp_start)
        games_per_min = (games_collected / elapsed) * 60.0
        logger.info(
            f"{variant}: ✓ Collected {games_collected} games "
            f"(avg {avg_moves:.0f} moves, {games_per_min:.1f} games/min)"
        )
        
        # Record throughput for adaptive tuning
        self.metrics.record_metric(f"{variant}_throughput", games_per_min, variant=variant)
        
        return games_collected
    
    def _train_one_step(self, variant: str) -> Optional[float]:
        """Perform a single training step for a variant.

        Returns the loss value, or None if the buffer is not ready.

        Interleaves puzzle + progame batches after the self-play batch when
        the corresponding toggles + counts are > 0. All batches share the
        same optimizer step (single forward+backward cycle per source per
        ``_train_one_step`` call).
        """
        buffer = self.buffers[variant]
        model = self.models[variant]
        optimizer = self.optimizers[variant]

        if not buffer.is_ready(min_size=self.BATCH_SIZE):
            return None

        try:
            positions, policies, values = buffer.sample(self.BATCH_SIZE)

            pos_tensor = torch.stack([torch.from_numpy(p).to(self.device) for p in positions])
            policy_tensor = torch.stack([torch.from_numpy(p).to(self.device) for p in policies])
            value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

            model.train()
            policy_logits, value_pred = model(pos_tensor)

            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_tensor)
            value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_tensor)
            loss = self.POLICY_LOSS_WEIGHT * policy_loss + self.VALUE_LOSS_WEIGHT * value_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Auxiliary puzzle batches (interleaved, same optimizer state)
            puzzle_total_loss = 0.0
            puzzle_batches_done = 0
            if self.PUZZLE_BATCHES_PER_GAME_BATCH > 0:
                for _ in range(self.PUZZLE_BATCHES_PER_GAME_BATCH):
                    pz = self.aux_loader.sample_puzzle_batch(self.BATCH_SIZE)
                    if pz is None:
                        break
                    pz_pos, pz_pol, pz_val = pz
                    pz_pos = pz_pos.to(self.device)
                    pz_pol = pz_pol.to(self.device)
                    pz_val = pz_val.to(self.device)
                    pz_logits, pz_vpred = model(pz_pos)
                    pz_policy_loss = torch.nn.functional.cross_entropy(pz_logits, pz_pol)
                    pz_value_loss = torch.nn.functional.mse_loss(pz_vpred.squeeze(), pz_val)
                    # Puzzle weighting: same POLICY_WEIGHT, lighter VALUE_WEIGHT
                    # (mated positions dominate value, so downweight)
                    pz_loss = self.POLICY_LOSS_WEIGHT * pz_policy_loss + 2.5 * pz_value_loss
                    pz_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    puzzle_total_loss += pz_loss.item()
                    puzzle_batches_done += 1

            # Auxiliary progame batches (interleaved, same optimizer state)
            pro_total_loss = 0.0
            pro_batches_done = 0
            if self.PROGAME_BATCHES_PER_GAME_BATCH > 0:
                for _ in range(self.PROGAME_BATCHES_PER_GAME_BATCH):
                    pg = self.aux_loader.sample_progame_batch(self.BATCH_SIZE)
                    if pg is None:
                        break
                    pg_pos, pg_pol, pg_val = pg
                    pg_pos = pg_pos.to(self.device)
                    pg_pol = pg_pol.to(self.device)
                    pg_val = pg_val.to(self.device)
                    pg_logits, pg_vpred = model(pg_pos)
                    pg_policy_loss = torch.nn.functional.cross_entropy(pg_logits, pg_pol)
                    pg_value_loss = torch.nn.functional.mse_loss(pg_vpred.squeeze(), pg_val)
                    pg_loss = self.POLICY_LOSS_WEIGHT * pg_policy_loss + self.VALUE_LOSS_WEIGHT * pg_value_loss
                    pg_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    pro_total_loss += pg_loss.item()
                    pro_batches_done += 1

            optimizer.step()

            # Step LR scheduler
            if variant in self.schedulers:
                self.schedulers[variant].step()

            self.total_training_steps += 1

            self.metrics.record_training_step(
                variant,
                loss.item(),
                policy_loss.item(),
                value_loss.item(),
                optimizer.param_groups[0]["lr"],
            )
            if puzzle_batches_done > 0:
                self.metrics.record_metric(
                    f"{variant}_puzzle_loss",
                    puzzle_total_loss / puzzle_batches_done,
                    variant=variant,
                )
            if pro_batches_done > 0:
                self.metrics.record_metric(
                    f"{variant}_progame_loss",
                    pro_total_loss / pro_batches_done,
                    variant=variant,
                )
            return loss.item()
        except Exception as e:
            logger.error(f"{variant} step failed: {e}", exc_info=True)
            return None

    def train_model(self, variant: str) -> float:
        """
        Train a model on its replay buffer.

        Args:
            variant: Model variant to train

        Returns:
            Average loss over training steps
        """
        losses = []
        for _ in range(self.TRAINING_STEPS_PER_ROUND):
            loss = self._train_one_step(variant)
            if loss is not None:
                losses.append(loss)
        if not losses:
            return None
        avg_loss = sum(losses) / len(losses)
        logger.info(f"{variant}: Training round complete, avg_loss={avg_loss:.4f}")
        return avg_loss

    def train_all_models_parallel(self):
        """
        Train all model variants.

        On CUDA: interleave training steps across variants to keep the GPU
        pipeline constantly fed (vs sequential per-variant which leaves the
        GPU idle ~60% of the time).

        On CPU: use ThreadPoolExecutor for parallelism.
        """
        if self.device.type == "cuda":
            # Interleaved: A1 B1 C1 A2 B2 C2 ... keeps GPU constantly busy
            results = {v: [] for v in self.VARIANTS}
            for step in range(self.TRAINING_STEPS_PER_ROUND):
                for variant in self.VARIANTS:
                    loss = self._train_one_step(variant)
                    if loss is not None:
                        results[variant].append(loss)

            avg_results = {}
            for variant in self.VARIANTS:
                losses = results[variant]
                if losses:
                    avg = sum(losses) / len(losses)
                    avg_results[variant] = avg
                    logger.info(f"✓ {variant}: training complete (loss={avg:.4f})")
                else:
                    avg_results[variant] = None
            return avg_results

        with ThreadPoolExecutor(max_workers=min(3, len(self.VARIANTS))) as executor:
            futures = {variant: executor.submit(self.train_model, variant) for variant in self.VARIANTS}
            results = {}
            for variant in self.VARIANTS:
                loss = futures[variant].result()
                results[variant] = loss
                if loss is not None:
                    logger.info(f"✓ {variant}: training complete (loss={loss:.4f})")
            return results
    
    def save_checkpoint(self, variant: str, step: int) -> str:
        """
        Save a model checkpoint.
        
        Args:
            variant: Model variant
            step: Step number for naming
        
        Returns:
            Path to saved checkpoint
        """
        model = self.models[variant]
        path = self.checkpoint_dir / f"{variant}_step_{step}.pt"
        
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizers[variant].state_dict() if variant in self.optimizers else None,
            "scheduler_state_dict": self.schedulers[variant].state_dict() if variant in self.schedulers else None,
            "config": self.model_configs[variant],
            "step": step,
            "variant": variant,
            "total_games": self.total_games,
            "total_training_steps": self.total_training_steps,
        }, path)
        
        logger.info(f"Saved checkpoint: {path}")
        
        # Register with evaluator (store path/metadata, avoid keeping full weights in RAM)
        self.evaluator.register_checkpoint(variant=variant, step=step, checkpoint_path=str(path))

        # Prune old checkpoints to save disk + prevent directory growth
        self._prune_checkpoints(variant)
        
        return str(path)
    
    def run(self, max_rounds: Optional[int] = None) -> None:
        """
        Main training loop.
        
        THE RULE:
        - Self-play generates all improvement (MCTS)
        - Training batches from replay buffer (purely from self-play)
        - Evaluation is low-frequency, only for monitoring
        
        Optimizations:
        - Adaptive MCTS visitation: automatically tunes visits to maintain target throughput
        - Disk management: prunes old buffers if disk space is low
        - RAM throttling: reduces workers if memory usage > 85%
        - GPU batching: if enabled, aggregates board evals from CPU workers (EXPERIMENTAL)
        
        Args:
            max_rounds: Maximum number of training rounds (None = infinite)
        """
        
        logger.info("Starting league training loop...")
        
        round_num = int(getattr(self, "start_round", 0))
        while max_rounds is None or round_num < max_rounds:
            self.round = round_num
            round_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num}")
            logger.info(f"{'='*60}")
            
            # Self-play phase (fully parallel across all variants)
            logger.info("Phase 1: Self-play generation (parallel across variants)...")
            sp_start = time.time()
            self._maybe_throttle_for_memory()
            with ThreadPoolExecutor(max_workers=self._variant_parallelism) as executor:
                # Submit all variants simultaneously
                futures = {}
                for variant in self.VARIANTS:
                    futures[variant] = executor.submit(self.generate_self_play, variant)
                
                # Collect results
                for variant in self.VARIANTS:
                    try:
                        games_collected = futures[variant].result()
                        logger.info(f"✓ {variant}: self-play complete ({games_collected} games)")
                    except Exception as e:
                        logger.error(f"✗ {variant}: self-play failed: {e}", exc_info=True)
                        raise
            logger.info(f"Phase 1 complete ({time.time()-sp_start:.1f}s)\n")
            
            # Training phase (GPU parallel with ThreadPoolExecutor)
            logger.info("Phase 2: Model training (parallel)...")
            tr_start = time.time()
            try:
                self.train_all_models_parallel()
            except Exception as e:
                logger.error(f"✗ Training phase failed: {e}", exc_info=True)
                return
            logger.info(f"Phase 2 complete ({time.time()-tr_start:.1f}s)\n")
            
            # Record buffer stats
            logger.info("Phase 3: Metrics collection...")
            for variant in self.VARIANTS:
                buffer_stats = self.buffers[variant].get_stats()
                logger.info(f"  {variant}: buffer {buffer_stats['size']}/{buffer_stats['capacity']} "
                           f"(mean_val={buffer_stats['value_mean']:.3f}±{buffer_stats['value_std']:.3f})")
                self.metrics.record_buffer_stats(
                    variant,
                    buffer_stats["size"],
                    buffer_stats["capacity"],
                    buffer_stats["value_mean"],
                    buffer_stats["value_std"],
                )
            
            # Checkpoint (less frequent)
            if (round_num + 1) % self.CHECKPOINT_EVERY_N_ROUNDS == 0:
                logger.info("Phase 4: Checkpointing...")
                for variant in self.VARIANTS:
                    try:
                        # Use 1-based step numbers for retention rules like step 15, 30, ...
                        self.save_checkpoint(variant, step=round_num + 1)
                    except Exception as e:
                        logger.error(f"Checkpoint failed for {variant}: {e}")

            # Replay buffer persistence (matches checkpoint cadence by default)
            if (round_num + 1) % self.BUFFER_SAVE_EVERY_N_ROUNDS == 0:
                for variant in self.VARIANTS:
                    try:
                        buffer_path = self.checkpoint_dir / f"{variant}_buffer_step_{round_num + 1}.npz"
                        self.buffers[variant].save_to_npz(str(buffer_path))
                    except Exception as e:
                        logger.warning(f"Buffer save failed for {variant}: {e}")
            
            # Check disk space periodically
            if (round_num + 1) % self.DISK_USAGE_CHECK_EVERY_N_ROUNDS == 0:
                self._check_and_manage_disk()
            
            # Evaluation (least frequent)
            if (round_num + 1) % self.EVAL_EVERY_N_ROUNDS == 0:
                logger.info("Phase 5: Evaluation...")
                self._run_evaluation_round(round_num)

            # Stockfish benchmark (configurable cadence; only if toggle is on)
            if (
                self.USE_STOCKFISH_EVAL
                and self.STOCKFISH_BENCH_EVERY_N_ROUNDS > 0
                and (round_num + 1) % self.STOCKFISH_BENCH_EVERY_N_ROUNDS == 0
            ):
                logger.info("Phase 6: Stockfish benchmark...")
                try:
                    self._run_stockfish_benchmark(round_num)
                except Exception as e:
                    logger.warning(f"Stockfish benchmark failed (non-fatal): {e}")
            
            # Metrics summary
            round_time = time.time() - round_start
            summary = self.metrics.get_summary()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ ROUND {round_num} COMPLETE ({round_time:.1f}s)")
            logger.info(f"{'='*60}")
            logger.info(f"Total games generated: {self.total_games}")
            logger.info(f"Total training steps: {self.total_training_steps}")
            logger.info("")
            
            # Per-variant stats
            for variant in self.VARIANTS:
                v_stats = summary.get("variants", {}).get(variant, {})
                logger.info(f"{variant.upper()}:")
                logger.info(f"  Games: {v_stats.get('games', 0)}")
                logger.info(f"  Train steps: {v_stats.get('train_steps', 0)}")
                logger.info(f"  Buffer: {v_stats.get('buffer_size', 0)}/{self.REPLAY_BUFFER_MAX_SIZE} "
                           f"({v_stats.get('buffer_fill_ratio', 0)*100:.1f}%)")
                if 'recent_loss' in v_stats:
                    logger.info(f"  Recent loss: {v_stats['recent_loss']:.4f}")
                if 'avg_game_length' in v_stats:
                    logger.info(f"  Avg game length: {v_stats['avg_game_length']:.1f} moves")
                logger.info("")
            
            logger.info(f"{'='*60}\n")
            self.metrics.log_summary(f"Round {round_num} complete ({round_time:.1f}s)")
            self.metrics.log_wandb_summary(summary, step=round_num)
            self._merge_gpu_stats_into_summary(summary)
            self.evolution_logger.append_round(
                round_num=round_num,
                summary=summary,
                round_time_seconds=round_time,
                note=(
                    f"games={self.total_games}, steps={self.total_training_steps}, "
                    f"mcts={self._current_mcts_visits}"
                ),
            )
            if (round_num + 1) % self.METRICS_EVERY_N_ROUNDS == 0:
                self.metrics.save_checkpoint(f"round_{round_num}")
            
            # Adaptive MCTS visitation tuning
            if (round_num + 1) % self.ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS == 0:
                self._adapt_mcts_visits()
            
            round_num += 1

        self.metrics.finish_wandb()

    def _maybe_throttle_for_memory(self) -> None:
        """Reduce self-play parallelism when RAM usage is high.

        Uses psutil if available; otherwise leaves settings unchanged.
        """
        try:
            import psutil

            mem = psutil.virtual_memory()
            used_pct = mem.percent

            # Restore defaults when memory pressure is low
            if used_pct < 80:
                self._num_self_play_workers = self.NUM_SELF_PLAY_WORKERS
                self._variant_parallelism = self.SELF_PLAY_VARIANT_PARALLELISM
                if self.use_gpu_batching:
                    # Keep parallelism = 3 (each variant has independent GPU process)
                    self._num_self_play_workers = self.GPU_SELF_PLAY_WORKERS
                self._buffer_target_size = self.REPLAY_BUFFER_MAX_SIZE
                for variant in self.VARIANTS:
                    buffer = self.buffers.get(variant)
                    if buffer is not None:
                        buffer.set_max_size(self._buffer_target_size)
                if self._last_buffer_target_size != self._buffer_target_size:
                    logger.info(f"RAM normal. Restored buffer size to {self._buffer_target_size}.")
                    self._last_buffer_target_size = self._buffer_target_size
                return

            # Moderate pressure: reduce workers and variant parallelism
            if used_pct >= 90:
                self._num_self_play_workers = max(1, self.NUM_SELF_PLAY_WORKERS // 2)
                self._variant_parallelism = 1
                self._buffer_target_size = max(10000, int(self.REPLAY_BUFFER_MAX_SIZE * 0.6))
            elif used_pct >= 85:
                self._num_self_play_workers = max(1, self.NUM_SELF_PLAY_WORKERS - 2)
                self._variant_parallelism = max(1, self.SELF_PLAY_VARIANT_PARALLELISM - 1)
                self._buffer_target_size = max(15000, int(self.REPLAY_BUFFER_MAX_SIZE * 0.8))

            # Keep GPU batching parallelism enabled (independent GPU processes per variant)

            logger.warning(
                f"High RAM usage ({used_pct:.1f}%). "
                f"Throttling self-play: workers={self._num_self_play_workers}, "
                f"variant_parallelism={self._variant_parallelism}"
            )

            for variant in self.VARIANTS:
                buffer = self.buffers.get(variant)
                if buffer is not None:
                    buffer.set_max_size(self._buffer_target_size)

            if self._last_buffer_target_size != self._buffer_target_size:
                logger.warning(f"Buffer size adjusted to {self._buffer_target_size} due to RAM usage.")
                self._last_buffer_target_size = self._buffer_target_size
        except Exception:
            # psutil not available or error; keep defaults
            self._num_self_play_workers = self.NUM_SELF_PLAY_WORKERS
            self._variant_parallelism = self.SELF_PLAY_VARIANT_PARALLELISM

    def _check_and_manage_disk(self) -> None:
        """Check disk space and prune buffer files if needed."""
        try:
            import shutil
            
            disk_usage = shutil.disk_usage(str(self.checkpoint_dir))
            free_pct = 100.0 * disk_usage.free / disk_usage.total
            free_gb = disk_usage.free / (1024**3)

            # Critical threshold: aggressively prune
            if free_pct < self.CRITICAL_DISK_THRESHOLD_PCT:
                logger.warning(
                    f"CRITICAL disk space: {free_pct:.1f}% free ({free_gb:.1f} GB). "
                    f"Purging old buffer files..."
                )
                self._aggressively_prune_buffer_files()
                # Also prune old checkpoints
                for variant in self.VARIANTS:
                    self._prune_checkpoints(variant)
                return

            # Moderate disk usage: keep only the most recent buffer files per variant
            for variant in self.VARIANTS:
                buffer_files = sorted(
                    self.checkpoint_dir.glob(f"{variant}_buffer_step_*.npz"),
                    key=lambda p: self._parse_step_from_buffer_file(p),
                    reverse=True  # Newest first
                )
                # Keep only the most recent N files
                if len(buffer_files) > self.MAX_BUFFER_FILES_PER_VARIANT:
                    for f in buffer_files[self.MAX_BUFFER_FILES_PER_VARIANT:]:
                        try:
                            f.unlink()
                            logger.info(f"Pruned old buffer file: {f.name}")
                        except Exception as e:
                            logger.warning(f"Failed to prune {f.name}: {e}")
                            
            logger.info(f"Disk usage: {free_pct:.1f}% free ({free_gb:.1f} GB)")
        except Exception as e:
            logger.warning(f"Disk check failed: {e}")

    def _aggressively_prune_buffer_files(self) -> None:
        """Delete all but the most recent buffer file per variant."""
        try:
            for variant in self.VARIANTS:
                buffer_files = sorted(
                    self.checkpoint_dir.glob(f"{variant}_buffer_step_*.npz"),
                    key=lambda p: self._parse_step_from_buffer_file(p),
                    reverse=True  # Newest first
                )
                # Keep only the very latest
                if len(buffer_files) > 1:
                    for f in buffer_files[1:]:
                        try:
                            f.unlink()
                            logger.warning(f"Aggressively pruned: {f.name}")
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Aggressive prune failed: {e}")

    def _parse_step_from_buffer_file(self, path: Path) -> int:
        """Extract step number from '<variant>_buffer_step_<step>.npz'."""
        name = path.name
        try:
            step_str = name.split("_")[-1].rstrip(".npz")
            return int(step_str)
        except (IndexError, ValueError):
            return 0

    def _adapt_mcts_visits(self) -> None:
        """
        Adjust MCTS visits based on recent throughput.
        
        Strategy:
        - If avg games/min < target: reduce visits (speed up)
        - If avg games/min > target: can afford to increase visits (quality)
        - Adjustments are gradual (±15%) and clamped [MIN, MAX]
        """
        try:
            # Collect recent throughput from metrics
            recent_throughputs = []
            for variant in self.VARIANTS:
                data = self.metrics.get_variant_throughput(variant)  # Returns games/min if available
                if data is not None and data > 0:
                    recent_throughputs.append(data)
            
            if not recent_throughputs:
                logger.debug("No throughput data available for adaptive tuning")
                return
            
            avg_throughput = sum(recent_throughputs) / len(recent_throughputs)
            
            # Compute adjustment
            if avg_throughput < self.TARGET_GAMES_PER_MINUTE * 0.9:
                # Below target: reduce visits for speed
                new_visits = max(
                    self.MIN_MCTS_VISITS,
                    int(self._current_mcts_visits * (1.0 - self.VISITS_ADJUSTMENT_FACTOR))
                )
                direction = "↓ (slower)"
            elif avg_throughput > self.TARGET_GAMES_PER_MINUTE * 1.1:
                # Above target: can increase visits for quality
                new_visits = min(
                    self.MAX_MCTS_VISITS,
                    int(self._current_mcts_visits * (1.0 + self.VISITS_ADJUSTMENT_FACTOR * 0.5))
                )
                direction = "↑ (better quality)"
            else:
                # On target: no adjustment
                new_visits = self._current_mcts_visits
                direction = "→ (on target)"
            
            if new_visits != self._current_mcts_visits:
                logger.info(
                    f"Adaptive MCTS: {avg_throughput:.1f} games/min (target {self.TARGET_GAMES_PER_MINUTE}). "
                    f"Adjusting visits {self._current_mcts_visits} → {new_visits} {direction}"
                )
                self._current_mcts_visits = new_visits
        except Exception as e:
            logger.debug(f"Adaptive tuning failed (non-critical): {e}")

    def _run_stockfish_benchmark(self, round_num: int) -> None:
        """Play short model-vs-Stockfish games and log a score / ELO estimate.

        Uses the baseline variant's current policy head with a one-argmax move
        selection (no MCTS) for speed. This is an approximation of true
        playing strength but is good enough to track progress over time.
        """
        from league.aux_phases import StockfishBenchmark

        if self._stockfish_benchmark is None:
            self._stockfish_benchmark = StockfishBenchmark(
                depth=self.STOCKFISH_DEPTH_BENCH,
                num_games=self.STOCKFISH_BENCH_NUM_GAMES,
                time_limit_ms=self.STOCKFISH_BENCH_TIME_LIMIT_MS,
            )

        baseline = self.models.get("baseline")
        if baseline is None:
            logger.warning("Stockfish benchmark skipped: no 'baseline' variant loaded")
            return

        baseline.eval()
        device = self.device

        def model_move_fn(board: chess.Board):
            import torch.nn.functional as F
            with torch.no_grad():
                from core.data import board_to_tensor
                tensor = board_to_tensor(board, 0, 22)
                inp = torch.tensor(tensor, dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = baseline(inp)
                # Mask illegal moves
                mask = torch.full_like(logits, float("-inf"))
                for m in board.legal_moves:
                    mask[0, m.from_square * 64 + m.to_square] = 0.0
                masked = logits + mask
                idx = int(masked.argmax(dim=-1).item())
                from_sq = idx // 64
                to_sq = idx % 64
                promo = None
                return chess.Move(from_sq, to_sq, promo)

        results = self._stockfish_benchmark.play_random_games(model_move_fn)
        self.metrics.set_gauge("stockfish_bench/score", results["score"], variant="baseline")
        self.metrics.set_gauge("stockfish_bench/elo_diff", results["elo_diff_estimate"], variant="baseline")
        self.evolution_logger.append_note(
            f"Stockfish bench round {round_num}: "
            f"{results['wins']}W-{results['draws']}D-{results['losses']}L "
            f"(score={results['score']:.2%}, elo_diff={results['elo_diff_estimate']:+.0f})"
        )
    
    def _run_evaluation_round(self, round_num: int) -> None:
        """
        Run a fair round-robin evaluation across the three variants.
        
        Args:
            round_num: Current round number
        """
        
        try:
            from league.fair_evaluation import build_fair_opening_fens

            opening_suite = build_fair_opening_fens()
            logger.info(
                f"Running fair round-robin evaluation on {len(opening_suite)} shared opening positions"
            )

            results = self.evaluator.compare_all_variants(
                models_by_variant=self.models,
                starting_fens=opening_suite,
            )

            scoreboard = results.get("scoreboard", {})
            for variant, stats in scoreboard.items():
                avg_score = float(stats.get("avg_score", 0.0))
                win_rate = float(stats.get("win_rate", 0.0))
                elo = float(stats.get("estimated_elo_diff", 0.0))

                self.metrics.set_gauge(f"{variant}_fair_score", avg_score, variant)
                self.metrics.set_gauge(f"{variant}_fair_win_rate", win_rate, variant)
                self.metrics.set_gauge(f"{variant}_fair_elo", elo, variant)

                logger.info(
                    f"FAIR EVAL {variant}: score={avg_score:.3f}, win_rate={win_rate:.1%}, elo≈{elo:.1f}"
                )

            self.evolution_logger.append_note(
                f"Fair evaluation round {round_num}: "
                + ", ".join(
                    f"{variant}={scoreboard[variant].get('estimated_elo_diff', 0.0):.1f} ELO"
                    for variant in self.VARIANTS
                    if variant in scoreboard
                )
            )

            for matchup, data in results.get("pairwise", {}).items():
                current_variant = data.get("current_variant", matchup.split("_vs_")[0])
                self.metrics.record_metric(
                    f"fair_eval/{matchup}/win_rate",
                    float(data.get("current_win_rate", 0.0)),
                    variant=current_variant,
                )
                self.metrics.record_metric(
                    f"fair_eval/{matchup}/elo",
                    float(data.get("estimated_elo_diff", 0.0)),
                    variant=current_variant,
                )

            self.metrics.save_checkpoint(f"fair_round_{round_num}")
            
        except Exception as e:
            logger.error(f"Evaluation round failed: {e}", exc_info=True)

    def _merge_gpu_stats_into_summary(self, summary: Dict[str, Any]) -> None:
        """Attach the latest GPU batching stats to the metrics summary for reporting."""
        variants = summary.setdefault("variants", {})

        for variant in self.VARIANTS:
            stats_path = self.log_dir / f"gpu_stats_{variant}.json"
            if not stats_path.exists():
                continue

            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            v_summary = variants.setdefault(variant, {})
            v_summary["gpu_avg_batch"] = float(stats.get("avg_batch_size", 0.0))
            v_summary["gpu_flush_size"] = int(stats.get("flush_by_size", 0))
            v_summary["gpu_flush_wait"] = int(stats.get("flush_by_wait", 0))
            v_summary["gpu_processed_evals"] = int(stats.get("processed_evals", 0))
            v_summary["gpu_total_batches"] = int(stats.get("total_batches", 0))
    
    def _model_constructor(self, **config):
        """
        Factory function for creating models.
        Override this or pass custom constructor.
        """
        # This would import from train.models
        raise NotImplementedError("Must provide model_constructor to LeagueTrainer")


if __name__ == "__main__":
    
    # Example usage
    trainer = LeagueTrainer(
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Initialize models (requires custom constructor)
    # trainer.initialize_models(model_constructor=...)
    
    # Run training loop
    # trainer.run(max_rounds=100)
