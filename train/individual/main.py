"""
Individual Training Entry Point
===============================

Single-threaded 3-phase curriculum training.

Usage:
    python -m individual.main --variant baseline
    python -m individual.main --variant attack --start-phase 2
    python -m individual.main --resume checkpoints_baseline/best.pt
"""

import os
import sys
import argparse
import multiprocessing as mp
from pathlib import Path

import torch

# Import from core
from core.models import create_model, load_model_with_compatibility
from core.data import PuzzleDataset, load_lichess_puzzles
from core.constants import MODEL_CONFIG, VALID_VARIANTS, HARDWARE_CONFIG
from core.utils import model_summary

# Import phases from this module
from .curriculum import phase1_puzzle_bootcamp, phase2_transition, phase3_pure_selfplay


def _setup_device():
    """Setup computation device with optimizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable TF32 for Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        torch.backends.cudnn.benchmark = True
    
    return device


def _get_generate_games_fn():
    """Get the game generation function from MCTS module."""
    try:
        from core.mcts import generate_mcts_game
        
        def generate_games(model, device, num_games, num_simulations):
            """Generate self-play games using MCTS."""
            games = []
            for i in range(num_games):
                if (i + 1) % 10 == 0:
                    print(f"  Game {i+1}/{num_games}")
                game = generate_mcts_game(
                    model=model,
                    device=device,
                    num_simulations=num_simulations
                )
                if game:
                    games.append(game)
            return games
        
        return generate_games
    except ImportError as e:
        print(f"⚠ MCTS not available: {e}")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="3-Phase Curriculum Chess AI Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m individual.main --variant baseline
  python -m individual.main --variant attack --start-phase 2
  python -m individual.main --resume checkpoints_baseline/phase1_best.pt
  python -m individual.main --skip-bootcamp --start-phase 1
        """
    )
    
    # Model selection
    parser.add_argument("--variant", default="baseline", choices=VALID_VARIANTS,
                        help="Model variant: baseline, attack, or est")
    
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: ./checkpoints_{variant})")
    
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # Phase control
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Start from phase (1=Puzzle Bootcamp, 2=Transition, 3=Self-Play)")
    
    parser.add_argument("--skip-bootcamp", action="store_true",
                        help="Skip checkmate bootcamp in Phase 1")
    
    args = parser.parse_args()
    
    # Set default checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints_{args.variant}"
    
    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Guard against overwriting league checkpoints
    project_root = Path(__file__).parent.parent
    league_checkpoint_dir = (project_root / "checkpoints").resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()

    if checkpoint_dir == league_checkpoint_dir or league_checkpoint_dir in checkpoint_dir.parents:
        raise SystemExit(
            f"Refusing to use league checkpoint directory for individual training: {checkpoint_dir}\n"
            f"Choose a different --checkpoint-dir (e.g., ./checkpoints_{args.variant})."
        )

    if args.resume:
        resume_path = Path(args.resume).resolve()
        if resume_path == league_checkpoint_dir or league_checkpoint_dir in resume_path.parents:
            raise SystemExit(
                f"Refusing to resume from league checkpoints: {resume_path}\n"
                "Copy the checkpoint to an individual directory before resuming."
            )
    
    print("\n" + "="*80)
    print("3-PHASE CURRICULUM CHESS AI TRAINING")
    print("="*80)
    print(f"Variant: {args.variant}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Starting phase: {args.start_phase}")
    print("="*80 + "\n")
    
    # Setup
    device = _setup_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create model
    print(f"Creating {args.variant} model...")
    model = create_model(variant=args.variant).to(device)
    model_summary(model)
    
    # Optional: Compile model for 2x speedup (PyTorch 2.0+)
    if HARDWARE_CONFIG.get('compile_model', False):
        try:
            model = torch.compile(model)
            print("✓ Model compiled with torch.compile\n")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}\n")
    
    # Resume from checkpoint
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        model = load_model_with_compatibility(model, args.resume, device)
        print("✓ Checkpoint loaded\n")
    
    # Get game generation function for self-play phases
    generate_games_fn = _get_generate_games_fn()
    
    # Load puzzle dataset for checkmate reinforcement
    puzzle_dataset = None
    if args.start_phase >= 3:
        try:
            puzzles = load_lichess_puzzles()
            puzzle_dataset = PuzzleDataset(
                puzzles=puzzles,
                input_channels=MODEL_CONFIG[args.variant]['input_channels']
            )
        except Exception as e:
            print(f"⚠ Could not load puzzles: {e}")
    
    # Run phases
    if args.start_phase <= 1:
        model = phase1_puzzle_bootcamp(
            model, args.variant, args.checkpoint_dir, 
            skip_bootcamp=args.skip_bootcamp
        )
    
    if args.start_phase <= 2:
        model = phase2_transition(
            model, args.variant, args.checkpoint_dir,
            generate_games_fn=generate_games_fn
        )
    
    if args.start_phase <= 3:
        model = phase3_pure_selfplay(
            model, args.variant, args.checkpoint_dir,
            generate_games_fn=generate_games_fn,
            puzzle_dataset=puzzle_dataset
        )
    
    print("\n✓ All phases complete!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
