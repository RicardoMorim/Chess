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

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from concurrent.futures import ThreadPoolExecutor
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
    
    # Training hyperparameters (BOOTSTRAP: minimal until flow validated)
    VARIANTS = ["baseline", "attack", "est"]
    # Parallelism config (INTERMEDIATE - validated and stable)
    # NOTE: Self-play already multiplies across variants; keep this modest to avoid oversubscription.
    NUM_SELF_PLAY_WORKERS = 2  # CPU processes per variant for self-play
    GAMES_PER_WORKER_PER_ROUND = 5  # Multiple games per worker
    BATCH_SIZE = 128  # Larger batches for GPU efficiency
    TRAINING_STEPS_PER_ROUND = 10  # More gradient updates per round
    CHECKPOINT_EVERY_N_ROUNDS = 5  # Checkpoint every 5 rounds
    EVAL_EVERY_N_ROUNDS = 100  # Skip for now

    # Devices / concurrency
    SELF_PLAY_DEVICE = "cpu"  # Safer: avoids many CUDA contexts across worker processes
    SELF_PLAY_VARIANT_PARALLELISM = 2  # How many variants generate self-play concurrently
    
    # MCTS hyperparameters (INTERMEDIATE - correct dual-budget approach)
    # Self-play: BALANCED generation of training data (GPU utilized 100%, VRAM 80%)
    MCTS_VISITS_SELFPLAY = 16  # Throughput-optimal; 32 doubled round time
    # Evaluation: SLOWER, higher quality comparisons between models
    MCTS_VISITS_EVAL = 64  # Meaningful search for model assessment
    C_PUCT = 4.0
    TEMPERATURE = 1.0
    DIRICHLET_ALPHA = 0.3
    
    # Replay buffer config
    # Large buffers can consume many GB of RAM (3 variants). Keep smaller unless you have ample RAM.
    REPLAY_BUFFER_MAX_SIZE = 50_000

    # Checkpoint retention
    CHECKPOINT_KEEP_LAST_N = 3
    CHECKPOINT_KEEP_EVERY_N = 15
    CHECKPOINT_ALWAYS_KEEP_STEPS = {1}
    
    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        device: str = "cuda",
    ):
        """
        Initialize league trainer.
        
        Args:
            checkpoint_dir: Directory for model checkpoints
            log_dir: Directory for logs and metrics
            device: Device for GPU training ("cuda" or "cpu")
        """
        self.device = torch.device(device)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize models, optimizers, buffers
        self.models = {}
        self.optimizers = {}
        self.buffers = {}
        self.model_configs = {}
        
        # Metrics and evaluation
        self.metrics = MetricsCollector(str(self.log_dir))
        self.evaluator = Evaluator(device=str(self.device))
        
        # State tracking
        self.round = 0
        self.start_round = 0
        self.total_games = 0
        self.total_training_steps = 0
        
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
            
            # Create model
            model = model_constructor(**config)
            model.to(self.device)
            
            # Create optimizer
            optimizer = optim.SGD(
                model.parameters(),
                lr=0.01,
                momentum=0.9,
                weight_decay=1e-4,
            )
            
            # Create replay buffer
            buffer = ReplayBuffer(max_size=self.REPLAY_BUFFER_MAX_SIZE)
            
            # Store
            self.models[variant] = model
            self.optimizers[variant] = optimizer
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
                    self.total_games = int(checkpoint.get("total_games", self.total_games))
                    self.total_training_steps = int(checkpoint.get("total_training_steps", self.total_training_steps))
                else:
                    state_dict = checkpoint
                    opt_state = None

                # Load model weights (non-strict for compatibility)
                self.models[variant].load_state_dict(state_dict, strict=False)

                # Load optimizer state if present
                if opt_state is not None and variant in self.optimizers:
                    try:
                        self.optimizers[variant].load_state_dict(opt_state)
                    except Exception as e:
                        logger.warning(f"{variant}: could not load optimizer state: {e}")

                logger.info(f"Resumed {variant} from checkpoint: {path.name}")
                max_step_loaded = max(max_step_loaded, int(step))

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
        
        model = self.models[variant]
        # Send CPU weights to workers to avoid CUDA tensor pickling and extra GPU memory pressure.
        model_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        
        # Launch workers
        for worker_id in range(self.NUM_SELF_PLAY_WORKERS):
            p = ctx.Process(
                target=self_play_worker,
                args=(
                    model_state,
                    self._model_constructor,
                    self.GAMES_PER_WORKER_PER_ROUND,
                    self.SELF_PLAY_DEVICE,
                    queue,
                    self.model_configs[variant],
                    {
                        "num_visits": self.MCTS_VISITS_SELFPLAY,  # Fast self-play
                        "temperature": self.TEMPERATURE,
                        "c_puct": self.C_PUCT,
                        "dirichlet_alpha": self.DIRICHLET_ALPHA,
                        "add_noise": True,
                        "parallel_workers": 1,
                    },
                    worker_id,
                ),
                name=f"{variant}_worker_{worker_id}",
            )
            p.start()
            processes.append(p)
        
        # Collect results
        num_games_expected = self.NUM_SELF_PLAY_WORKERS * self.GAMES_PER_WORKER_PER_ROUND
        games_collected = 0
        errors = 0
        game_lengths = []
        
        logger.info(f"{variant}: Waiting for {num_games_expected} games from {self.NUM_SELF_PLAY_WORKERS} workers...")
        
        for _ in range(num_games_expected):
            result = queue.get()
            
            if "error" in result:
                logger.error(f"Worker {result.get('worker_id', '?')} error: {result['error']}")
                errors += 1
                continue
            
            game_data = result["game_data"]
            self.buffers[variant].add_game(game_data)
            
            games_collected += 1
            self.total_games += 1
            game_lengths.append(len(game_data))
            
            # Record metrics
            game_length = len(game_data)
            outcome = "1-0"  # Would need to extract from game_data
            self.metrics.record_self_play_game(variant, game_length, outcome)
            
            # Progress indicator
            if games_collected % 5 == 0 or games_collected == num_games_expected:
                logger.info(f"{variant}: Collected {games_collected}/{num_games_expected} games...")
        
        # Wait for workers to finish
        for p in processes:
            p.join()
        
        if errors > 0:
            logger.warning(f"{variant}: {errors}/{num_games_expected} games failed")
        
        avg_moves = sum(game_lengths) / len(game_lengths) if game_lengths else 0
        logger.info(f"{variant}: ✓ Collected {games_collected} games (avg {avg_moves:.0f} moves)")
        return games_collected
    
    def train_model(self, variant: str) -> float:
        """
        Train a model on its replay buffer.
        
        Args:
            variant: Model variant to train
        
        Returns:
            Average loss over training steps
        """
        
        buffer = self.buffers[variant]
        model = self.models[variant]
        optimizer = self.optimizers[variant]
        
        if not buffer.is_ready(min_size=self.BATCH_SIZE):
            logger.info(f"{variant}: Buffer not ready ({len(buffer)} < {self.BATCH_SIZE})")
            return None
        
        model.train()
        total_loss = 0.0
        
        for step in range(self.TRAINING_STEPS_PER_ROUND):
            try:
                positions, policies, values = buffer.sample(self.BATCH_SIZE)
                
                # Convert to tensors
                pos_tensor = torch.stack([torch.from_numpy(p).to(self.device) for p in positions])
                policy_tensor = torch.stack([torch.from_numpy(p).to(self.device) for p in policies])
                value_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
                
                # Forward pass
                policy_logits, value_pred = model(pos_tensor)
                
                # Compute losses
                policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_tensor)
                value_loss = torch.nn.functional.mse_loss(value_pred.squeeze(), value_tensor)
                
                # Combine losses
                loss = policy_loss + value_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                self.total_training_steps += 1
                
                # Record metrics
                self.metrics.record_training_step(
                    variant,
                    loss.item(),
                    policy_loss.item(),
                    value_loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
            
            except Exception as e:
                logger.error(f"{variant} training step {step} failed: {e}", exc_info=True)
        
        avg_loss = total_loss / self.TRAINING_STEPS_PER_ROUND
        model.eval()
        
        logger.info(f"{variant}: Training round complete, avg_loss={avg_loss:.4f}")
        return avg_loss
    
    def train_all_models_parallel(self):
        """
        Train all model variants in parallel using ThreadPoolExecutor.
        Reduces wall-clock time by ~2.5x (GPU utilization: ~40% → ~85%).
        
        Each thread trains one variant on GPU (non-blocking). PyTorch handles
        the GPU scheduling across threads safely.
        """
        # On a single GPU, parallel training threads can increase VRAM fragmentation and reduce stability.
        # Keep it sequential on CUDA; allow parallelism on CPU.
        if self.device.type == "cuda":
            results = {}
            for variant in self.VARIANTS:
                loss = self.train_model(variant)
                results[variant] = loss
                if loss is not None:
                    logger.info(f"✓ {variant}: training complete (loss={loss:.4f})")
            return results

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
            with ThreadPoolExecutor(max_workers=self.SELF_PLAY_VARIANT_PARALLELISM) as executor:
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
            
            # Evaluation (least frequent)
            if (round_num + 1) % self.EVAL_EVERY_N_ROUNDS == 0:
                logger.info("Phase 5: Evaluation...")
                self._run_evaluation_round(round_num)
            
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
            self.metrics.save_checkpoint(f"round_{round_num}")
            
            round_num += 1
    
    def _run_evaluation_round(self, round_num: int) -> None:
        """
        Run low-frequency evaluation against baseline.
        
        Args:
            round_num: Current round number
        """
        
        try:
            # Check if we have a baseline checkpoint to test against
            baseline_ckpts = list(self.checkpoint_dir.glob("baseline_step_*.pt"))
            
            if not baseline_ckpts:
                logger.warning("No baseline checkpoints found for evaluation")
                return
            
            # Use most recent baseline checkpoint
            latest_baseline = sorted(baseline_ckpts)[-1]
            logger.info(f"Evaluating against baseline: {latest_baseline}")
            
            # Load and evaluate (would need to implement full evaluation logic)
            
        except Exception as e:
            logger.error(f"Evaluation round failed: {e}", exc_info=True)
    
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
