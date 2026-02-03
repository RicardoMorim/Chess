#!/usr/bin/env python3
"""
League Training Main Entry Point

Starts the parallelized self-play + training loop.

This replaces the old single-threaded training system with:
- Parallel self-play workers (CPU)
- Persistent replay buffers (per-model)
- Batched GPU training
- Automatic checkpointing
- Low-frequency evaluation
- Centralized metrics collection

Run:
    python league/main.py

Monitor:
    tail -f logs/metrics.log

The Rule: Only MCTS self-play improves models. Everything else measures or protects.
"""

import sys
import torch
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from league.league_trainer import LeagueTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/league_training.log"),
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main training entry point."""
    
    logger.info("="*60)
    logger.info("LEAGUE TRAINING SYSTEM - STARTING UP")
    logger.info("="*60)
    
    # Device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = LeagueTrainer(
        checkpoint_dir="/checkpoints_league",
        log_dir="/logs",
        device=device,
    )
    
    # Import model constructor
    try:
        from core import create_model
        logger.info("Loaded model constructor from core")
    except ImportError as e:
        logger.error(f"Failed to import model: {e}")
        logger.error("Make sure core/__init__.py exists and exports create_model()")
        return 1
    
    # Initialize models with different configurations
    logger.info("Initializing models...")
    try:
        trainer.initialize_models(
            model_constructor=create_model,
            model_configs={
                "baseline": {
                    "num_blocks": 15,
                    "channels": 256,
                },
                "attack": {
                    "num_blocks": 15,
                    "channels": 256,
                },
                "est": {
                    "channels": 256,
                    "shared_blocks": 5,
                    "policy_blocks": 5,
                    "value_blocks": 5,
                },
            }
        )
        logger.info("Models initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize models: {e}", exc_info=True)
        return 1
    
    # Log configuration
    logger.info("\n" + "="*60)
    logger.info("TRAINING CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Variants: {trainer.VARIANTS}")
    logger.info(f"Self-play workers: {trainer.NUM_SELF_PLAY_WORKERS}")
    logger.info(f"Games per worker: {trainer.GAMES_PER_WORKER_PER_ROUND}")
    logger.info(f"Batch size: {trainer.BATCH_SIZE}")
    logger.info(f"Training steps per round: {trainer.TRAINING_STEPS_PER_ROUND}")
    logger.info(f"Checkpoint every N rounds: {trainer.CHECKPOINT_EVERY_N_ROUNDS}")
    logger.info(f"Evaluate every N rounds: {trainer.EVAL_EVERY_N_ROUNDS}")
    logger.info(f"MCTS visits (self-play): {trainer.MCTS_VISITS_SELFPLAY}")
    logger.info(f"MCTS visits (evaluation): {trainer.MCTS_VISITS_EVAL}")
    logger.info(f"Temperature: {trainer.TEMPERATURE}")
    logger.info("="*60 + "\n")
    
    # Run training loop
    logger.info("Starting main training loop...")
    logger.info("Press Ctrl+C to stop gracefully (will save metrics)")
    
    try:
        trainer.run(max_rounds=None)
    except KeyboardInterrupt:
        logger.info("\n" + "="*60)
        logger.info("TRAINING INTERRUPTED BY USER")
        logger.info("="*60)
        logger.info("Saving final metrics...")
        try:
            trainer.metrics.save_checkpoint("interrupted")
            logger.info("Final metrics saved")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
        
        logger.info("Goodbye!")
        return 0
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
