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

# Ensure `train/` is on sys.path so `core` imports work when launched from repo root.
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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


def _get_generate_games_fn(*, variant: str, selfplay_workers: int, mcts_parallel_workers: int):
    """Get the game generation function used by Phase 2/3.

    Uses CPU-parallel self-play by default to utilize available cores without
    creating many CUDA contexts.
    """
    try:
        from individual.selfplay_parallel import generate_games_parallel_from_state

        def generate_games(*, model=None, model_state_dict=None, device=None, num_games=0, num_simulations=0):
            # Self-play is intentionally CPU-based; `device` is kept for interface compatibility.
            if model_state_dict is None:
                if model is None:
                    raise ValueError("generate_games requires either model or model_state_dict")
                model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            return generate_games_parallel_from_state(
                variant=variant,
                model_state_dict=model_state_dict,
                num_games=int(num_games),
                num_simulations=int(num_simulations),
                temperature=1.0,
                selfplay_workers=int(selfplay_workers),
                mcts_parallel_workers=int(mcts_parallel_workers),
            )

        return generate_games

    except Exception as e:
        print(f"⚠ Parallel self-play not available: {e}")
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

    # Performance / safety knobs (Phase 2/3 self-play)
    parser.add_argument("--selfplay-workers", type=int, default=None,
                        help="CPU worker processes for self-play (default: HARDWARE_CONFIG['selfplay_workers'] or 8)")
    parser.add_argument("--mcts-parallel-workers", type=int, default=None,
                        help="Threads per worker for MCTS simulations (default: 1). Keep low when self-play is multi-process.")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="Max Phase 3 iterations (default: infinite)")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save a Phase 3 checkpoint every N iterations (default: 1)")
    parser.add_argument("--keep-last", type=int, default=3,
                        help="Checkpoint retention: keep last N phase3_iter checkpoints (default: 3)")
    parser.add_argument("--keep-every", type=int, default=15,
                        help="Checkpoint retention: keep every Nth phase3_iter checkpoint (default: 15)")
    
    args = parser.parse_args()
    
    # Set default checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints_{args.variant}"
    
    return args


def main():
    """Main entry point."""
    args = parse_args()

    # Guard against overwriting league checkpoints
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
    
    # Auto-resume Phase 3 if possible (unless --resume explicitly provided)
    if args.resume is None and args.start_phase == 3:
        ckpt_dir = Path(args.checkpoint_dir)
        candidates = sorted(ckpt_dir.glob("phase3_iter_*.pt"))
        if candidates:
            # Pick highest iteration from filenames
            def _iter_num(p: Path):
                name = p.name
                try:
                    return int(name[len("phase3_iter_"):-len(".pt")])
                except Exception:
                    return -1

            latest = sorted(candidates, key=_iter_num)[-1]
            if _iter_num(latest) > 0:
                args.resume = str(latest)
                print(f"Auto-resume: found latest Phase 3 checkpoint: {latest}")
        else:
            interrupted = ckpt_dir / "phase3_interrupted.pt"
            if interrupted.exists():
                args.resume = str(interrupted)
                print(f"Auto-resume: found interrupted checkpoint: {interrupted}")

    # Resume from checkpoint (model weights)
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        model = load_model_with_compatibility(model, args.resume, device)
        print("✓ Checkpoint loaded\n")
    
    # Self-play parallelism defaults
    if args.selfplay_workers is None:
        args.selfplay_workers = int(HARDWARE_CONFIG.get('selfplay_workers', 8))
    if args.mcts_parallel_workers is None:
        # Important: when using many processes, keep per-process MCTS threads low.
        args.mcts_parallel_workers = 1

    # Get game generation function for self-play phases
    generate_games_fn = _get_generate_games_fn(
        variant=args.variant,
        selfplay_workers=args.selfplay_workers,
        mcts_parallel_workers=args.mcts_parallel_workers,
    )
    
    # Load puzzle dataset for checkmate reinforcement
    puzzle_dataset = None
    if args.start_phase >= 3:
        try:
            puzzles = load_lichess_puzzles()
            input_channels = MODEL_CONFIG[args.variant]["input_channels"]
            model_type = "big" if input_channels >= 22 else ("medium" if input_channels >= 20 else "small")
            puzzle_dataset = PuzzleDataset(
                puzzles=puzzles,
                model_type=model_type,
                cache_dir=str(Path(args.checkpoint_dir) / "cache"),
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
            puzzle_dataset=puzzle_dataset,
            max_iterations=args.max_iterations,
            save_every=max(1, int(args.save_every)),
            keep_last=max(0, int(args.keep_last)),
            keep_every=max(0, int(args.keep_every)),
            resume_path=args.resume,
        )
    
    print("\n✓ All phases complete!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
