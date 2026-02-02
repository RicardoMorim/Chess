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
    NUM_SELF_PLAY_WORKERS = 1  # Start minimal
    GAMES_PER_WORKER_PER_ROUND = 1  # Validate flow first
    BATCH_SIZE = 64  # Smaller for faster feedback
    TRAINING_STEPS_PER_ROUND = 5  # Validate before scaling
    CHECKPOINT_EVERY_N_ROUNDS = 1  # Frequent saves
    EVAL_EVERY_N_ROUNDS = 100  # Skip for now
    
    # MCTS hyperparameters (DRASTICALLY REDUCED for bootstrap)
    MCTS_VISITS_TRAINING = 16  # Minimal for validation
    MCTS_VISITS_EVAL = 32
    C_PUCT = 4.0
    TEMPERATURE = 1.0
    DIRICHLET_ALPHA = 0.3
    
    # Replay buffer config
    REPLAY_BUFFER_MAX_SIZE = 200_000
    
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
        self.total_games = 0
        self.total_training_steps = 0
        
        logger.info(f"LeagueTrainer initialized: device={self.device}, checkpoints={self.checkpoint_dir}")
    
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
        model_state = model.state_dict()
        
        # Launch workers
        for worker_id in range(self.NUM_SELF_PLAY_WORKERS):
            p = ctx.Process(
                target=self_play_worker,
                args=(
                    model_state,
                    self._model_constructor,
                    self.GAMES_PER_WORKER_PER_ROUND,
                    "cpu",
                    queue,
                    self.model_configs[variant],
                    {
                        "num_visits": self.MCTS_VISITS_TRAINING,
                        "temperature": self.TEMPERATURE,
                        "c_puct": self.C_PUCT,
                        "dirichlet_alpha": self.DIRICHLET_ALPHA,
                        "add_noise": True,
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
        
        for _ in range(num_games_expected):
            result = queue.get()
            
            if "error" in result:
                logger.error(f"Worker error: {result['error']}")
                continue
            
            game_data = result["game_data"]
            self.buffers[variant].add_game(game_data)
            
            games_collected += 1
            self.total_games += 1
            
            # Record metrics
            game_length = len(game_data)
            outcome = "1-0"  # Would need to extract from game_data
            self.metrics.record_self_play_game(variant, game_length, outcome)
        
        # Wait for workers to finish
        for p in processes:
            p.join()
        
        logger.info(f"{variant}: Collected {games_collected} games")
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
            "config": self.model_configs[variant],
            "step": step,
            "variant": variant,
        }, path)
        
        logger.info(f"Saved checkpoint: {path}")
        
        # Register with evaluator
        self.evaluator.register_checkpoint(variant, step, model.state_dict())
        
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
        
        round_num = 0
        while max_rounds is None or round_num < max_rounds:
            self.round = round_num
            round_start = time.time()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"ROUND {round_num}")
            logger.info(f"{'='*60}")
            
            # Self-play phase (CPU parallel)
            logger.info("Phase 1: Self-play generation...")
            sp_start = time.time()
            for variant in self.VARIANTS:
                try:
                    self.generate_self_play(variant)
                    logger.info(f"✓ {variant}: self-play complete")
                except Exception as e:
                    logger.error(f"✗ {variant}: self-play failed: {e}", exc_info=True)
                    return
            logger.info(f"Phase 1 complete ({time.time()-sp_start:.1f}s)\n")
            
            # Training phase (GPU sequential, batched)
            logger.info("Phase 2: Model training...")
            tr_start = time.time()
            for variant in self.VARIANTS:
                try:
                    self.train_model(variant)
                    logger.info(f"✓ {variant}: training complete")
                except Exception as e:
                    logger.error(f"✗ {variant}: training failed: {e}", exc_info=True)
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
                        self.save_checkpoint(variant, round_num)
                    except Exception as e:
                        logger.error(f"Checkpoint failed for {variant}: {e}")
            
            # Evaluation (least frequent)
            if (round_num + 1) % self.EVAL_EVERY_N_ROUNDS == 0:
                logger.info("Phase 5: Evaluation...")
                self._run_evaluation_round(round_num)
            
            # Metrics summary
            round_time = time.time() - round_start
            logger.info(f"\n{'='*60}")
            logger.info(f"✓ ROUND {round_num} COMPLETE ({round_time:.1f}s)")
            logger.info(f"Total games: {self.total_games}")
            logger.info(f"Total training steps: {self.total_training_steps}")
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
