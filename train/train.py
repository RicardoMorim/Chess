import time
import torch
from torch.utils.data import DataLoader
import os
import glob
import json
import signal
import sys
import gc
import random
import argparse


from models import ChessNet, create_chess_model, load_model_with_compatibility
from data import (ChessDataset, PuzzleDataset, load_puzzles, load_lichess_puzzles, 
                 filter_and_prioritize_puzzles_cached, load_professional_games, 
                 load_games_in_batches)
from utils import clear_memory, test_tactical_recognition, get_optimal_batch_size, model_summary
from self_play import generate_self_play_games, run_self_play_training
from training import train_batch, train_tactical

# Import optimizations
try:
    from optimizations import (
        HARDWARE_CONFIG, 
        create_optimized_dataloader,
        AsyncDataPrefetcher,
        aggressive_memory_cleanup,
        print_memory_stats
    )
    HAS_OPTIMIZATIONS = True
    print("✓ Training optimizations loaded")
except ImportError:
    HAS_OPTIMIZATIONS = False
    print("⚠ Optimizations not available, using defaults")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable TF32 for faster training on Ampere+ GPUs (no effect on Pascal/GTX 1050)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True


# Signal handler 
def signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch):
    print("\nTraining interrupted! Saving model and state...")
    torch.save(model.state_dict(), save_path)
    with open(state_file, 'w') as f:
        state = {"processed_games": processed_games, "last_epoch": current_epoch + 1}
        json.dump(state, f)
    print(f"Model saved to {save_path}, state saved to {state_file}")
    sys.exit(0)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chess AI Training Script (Improved)")
    
    # Training mode
    parser.add_argument("mode", nargs="?", default=None, choices=["pro", "regular", "self-play"], 
                        help="Training mode: professional games, regular games, or self-play")
    
    # Model parameters
    parser.add_argument("--model", default="big", choices=["limited", "small", "medium", "big"], 
                        help="Model size: limited (low-VRAM), small (6 blocks), medium (10 blocks), or big (15 blocks)")
    
    parser.add_argument("--model-path", default=None, 
                        help="Path to save/load the model (default: ./chess_model/[model_size]_model.pth)")
    
    parser.add_argument("--legacy", action="store_true",
                        help="Load legacy model architecture (for old checkpoints)")
    
    parser.add_argument("--no-se", action="store_true",
                        help="Disable Squeeze-and-Excitation blocks")
    
    # MCTS parameters
    parser.add_argument("--no-mcts", action="store_true", 
                        help="Disable MCTS for self-play (faster but lower quality)")
    
    parser.add_argument("--fast-mcts", action="store_true", 
                        help="Use fast MCTS for self-play (balanced speed/quality)")
    
    # Self-play parameters when in self-play mode
    parser.add_argument("games", nargs="?", type=int, default=None,
                        help="Number of games per batch in self-play mode")
    
    parser.add_argument("iterations", nargs="?", type=int, default=None,
                        help="Number of iterations per cycle in self-play mode")
    
    args = parser.parse_args()
    
    # Set defaults based on arguments
    if args.model_path is None:
        args.model_path = f"./chess_model/{args.model}_model.pth"
        
    # Backward compatibility for positional arguments
    if len(sys.argv) > 1 and sys.argv[1] in ["pro", "regular", "self-play"] and args.mode is None:
        args.mode = sys.argv[1]
    
    return args


# Main execution function
def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print(f"\n{'='*60}")
    print("CHESS AI TRAINING")
    print(f"{'='*60}")
    
    # Initialize model based on size
    use_se = not args.no_se
    
    if args.model == "limited":
        print(f"Model: Limited (low-VRAM, 4 blocks, 64 filters, 18 channels, SE={use_se})")
        model = create_chess_model("limited", use_se=use_se, legacy=args.legacy).to(device)
    elif args.model == "small":
        print(f"Model: Small (6 blocks, 18 channels, SE={use_se})")
        model = create_chess_model("small", use_se=use_se, legacy=args.legacy).to(device)
    elif args.model == "medium":
        print(f"Model: Medium (10 blocks, 22 channels, SE={use_se})")
        model = create_chess_model("medium", use_se=use_se, legacy=args.legacy).to(device)
    else:  # "big"
        print(f"Model: Big (15 blocks, 22 channels, SE={use_se})")
        model = create_chess_model("big", use_se=use_se, legacy=args.legacy).to(device)
    
    # Print model summary
    model_summary(model)
        
    # Setup paths
    save_path = args.model_path
    model_dir = os.path.dirname(save_path)
    state_file = os.path.join(model_dir, "training_state.json")
    pro_state_file = os.path.join(model_dir, "pro_training_state.json")
    
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Determine MCTS settings
    use_mcts = not args.no_mcts
    fast_mcts = args.fast_mcts
    
    if not use_mcts:
        print("MCTS disabled for self-play (faster but lower quality)")
    elif fast_mcts:
        print("Using fast MCTS for self-play (balanced speed/quality)")
    else:
        print("Full MCTS enabled for self-play (highest quality games)")
    
    # Load existing model if available
    if os.path.exists(save_path):
        print(f"Loading existing model from {save_path}")
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded existing model from {save_path}")

    # Get training state
    current_epoch = 0
    processed_games = 0
    pro_game_count = 0
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_games = state.get("processed_games", 0)
            current_epoch = state.get("last_epoch", 0)
    
    if os.path.exists(pro_state_file):
        with open(pro_state_file, 'r') as f:
            pro_state = json.load(f)
            pro_files_remaining = pro_state.get("current_pro_file_idx", 0) < len(glob.glob(os.path.join("./chess_pgns/pros", "*.pgn")))
            pro_game_count = pro_state.get("processed_pro_games", 0)
            
            # If we've already processed all pro files, start with regular games
            if not pro_files_remaining:
                print("All professional games have been processed. Starting with regular games.")
                current_phase = "regular"
            else:
                current_phase = "professional"
    else:
        current_phase = "professional"  # Start with professional games by default
        pro_state = {}
        pro_game_count = 0

    # Set up signal handlers
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))

    # Get file paths
    pgn_directory = "./chess_pgns"
    pgn_files = glob.glob(os.path.join(pgn_directory, "*.pgn"))

    # Load puzzles once - they're small enough
    puzzle_pgn = "./chess_pgns/puzzles/puzzles.pgn"
    lichess_csv = "./chess_pgns/puzzles/lichess_db_puzzle.csv"
    
    # Only try to load if files exist
    pgn_puzzles = []
    if os.path.exists(puzzle_pgn):
        pgn_puzzles = load_puzzles(puzzle_pgn)
        print(f"Loaded {len(pgn_puzzles)} PGN puzzles")
    
    lichess_puzzles = []  
    if os.path.exists(lichess_csv):
        lichess_puzzles = load_lichess_puzzles(lichess_csv)
        print(f"Loaded {len(lichess_puzzles)} Lichess puzzles")

    all_puzzles = pgn_puzzles + lichess_puzzles
    prioritized_puzzles = filter_and_prioritize_puzzles_cached(all_puzzles)
    
    # Add this line to determine model type based on the loaded model
    model_type = "small" if model.is_small_model() else "big"
    
    # Pass model_type to puzzle_dataset to ensure matching channels
    puzzle_dataset = PuzzleDataset(prioritized_puzzles, model_type=model_type)
    print(f"Total puzzles after prioritization: {len(prioritized_puzzles)}")
    print(f"Using {model_type} model architecture with {model.input_channels} input channels")

    # Find optimal batch size
    print("Determining optimal batch size...")
    model.eval()
    optimal_batch_size = get_optimal_batch_size(model, device, starting_size=64)
    model.train()
    print(f"Using optimal batch size: {optimal_batch_size}")
    
    
    # Create puzzle dataloader once - puzzles are smaller and reused
    # Use optimized dataloader if available
    if HAS_OPTIMIZATIONS:
        puzzle_dataloader = create_optimized_dataloader(
            puzzle_dataset,
            batch_size=min(32, optimal_batch_size),
            shuffle=True,
            for_training=True
        )
        print(f"Using optimized DataLoader (workers={HARDWARE_CONFIG['dataloader_workers']})")
    else:
        puzzle_dataloader = DataLoader(
            puzzle_dataset, 
            batch_size=min(32, optimal_batch_size),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
    
    # Determine training mode
    current_phase = "regular"  # Default mode
    is_mode_locked = False  # Track if mode is locked by command line
    
    # Set mode based on command line argument
    if args.mode:
        current_phase = args.mode
        is_mode_locked = True
        print(f"Command-line override: Using {current_phase} training mode (locked)")
            
    # Set batch sizes - smaller batch sizes for faster processing
    pro_batch_size = 1000
    regular_batch_size = 1000
    
    # Main training loop
    max_iterations = 1000000  # Very high limit (essentially unlimited)
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        print(f"\n--- Training Iteration {iterations} ---")
        
        # Self-play mode
        if current_phase == "self-play":
            print("\n=== SELF-PLAY REINFORCEMENT LEARNING MODE ===")
            
            # Default number of self-play games and iterations for continuous mode
            games_per_batch = 30  # Lower default for continuous mode
            iterations_per_cycle = 2  # Fewer iterations per cycle for faster feedback
            
            # Use command line arguments if provided
            if args.games is not None:
                games_per_batch = args.games
                
            if args.iterations is not None:
                iterations_per_cycle = args.iterations
                
            print(f"Running {iterations_per_cycle} iterations with {games_per_batch} games per iteration")
            
            # Run self-play training for this batch
            model = run_self_play_training(
                model, 
                device,
                save_path, 
                state_file, 
                num_games=games_per_batch,
                num_iterations=iterations_per_cycle,
                use_mcts=use_mcts,
                fast_mcts=fast_mcts
            )
            
            # Run tactical training after self-play to maintain tactical awareness
            print("Running tactical training phase...")
            tactical_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            tactical_epochs = min(6 + iterations // 5, 20)  # Start with 6, gradually increase
            train_tactical(model, tactical_optimizer, puzzle_dataloader, device, epochs=tactical_epochs)
            
            # Test tactical recognition occasionally
            if iterations % 3 == 0:
                test_accuracy = test_tactical_recognition(model, device)
                print(f"Tactical recognition accuracy: {test_accuracy:.2%}")
            
            # Continue with self-play mode if locked, otherwise potentially switch
            if not is_mode_locked and iterations % 10 == 0:
                # Occasionally switch to other modes for variety
                current_phase = random.choice(["regular", "pro", "self-play"])
                print(f"Switching to {current_phase} mode for variety")
                
        # Professional games mode
        elif current_phase == "pro":
            print("\n=== PROFESSIONAL GAMES TRAINING ===")
            
            # Load one batch of professional games
            pro_games = load_professional_games(pro_state_file, batch_size=pro_batch_size)
            
            if not pro_games:
                print("No more professional games to process.")
                if is_mode_locked:
                    print("Mode is locked to 'pro' but no more pro games available.")
                    print("Will attempt to reload pro games in the next iteration.")
                    time.sleep(5)  # Wait before trying again
                    continue
                else:
                    print("Switching to regular games mode temporarily.")
                    current_phase = "regular"
                    continue
            
            batch_size = len(pro_games)
            print(f"Processing professional batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(pro_games, augment=True, model_type=model_type)
            
            if HAS_OPTIMIZATIONS:
                game_dataloader = create_optimized_dataloader(
                    game_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    for_training=True
                )
            else:
                game_dataloader = DataLoader(
                    game_dataset, 
                    batch_size=optimal_batch_size,
                    shuffle=True, 
                    num_workers=min(2, os.cpu_count() or 1),
                    pin_memory=True
                )
            
            # Train on this batch
            train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, 
                    epochs=5, processed_games=processed_games, device=device)
            
            # Clean up to free memory before next phase
            del pro_games
            del game_dataset
            del game_dataloader
            gc.collect()
            if HAS_OPTIMIZATIONS:
                aggressive_memory_cleanup()
            else:
                clear_memory()
            
            # Tactical training after professional batch
            print("Running quick tactical training phase...")
            tactical_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            train_tactical(model, tactical_optimizer, puzzle_dataloader, device, epochs=6)
            
            
            # Test tactical recognition occasionally
            if iterations % 3 == 0:
                test_accuracy = test_tactical_recognition(model, device)
                print(f"Tactical recognition accuracy: {test_accuracy:.2%}")
            
            # Only switch modes if not locked
            if not is_mode_locked and iterations % 5 == 0:
                current_phase = "regular"
                print("Switching to regular games mode for variety")
                
        # Regular games mode
        else:  # Regular games phase
            print("\n=== REGULAR GAMES TRAINING ===")
            if HAS_OPTIMIZATIONS:
                print_memory_stats("before loading")
            
            # Load one batch of regular games
            regular_games = load_games_in_batches(pgn_files, state_file, batch_size=regular_batch_size)
            
            if not regular_games:
                print("No regular games available or error loading games.")
                if is_mode_locked:
                    print("Mode is locked to 'regular' but having trouble loading games.")
                    print("Will attempt to reload games in the next iteration.")
                    time.sleep(5)  # Wait before trying again
                    continue
                else:
                    print("Switching to professional games mode temporarily.")
                    current_phase = "pro"
                    continue
                
            batch_size = len(regular_games)
            print(f"Processing regular batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(regular_games, augment=True)
            
            if HAS_OPTIMIZATIONS:
                game_dataloader = create_optimized_dataloader(
                    game_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    for_training=True
                )
            else:
                game_dataloader = DataLoader(
                    game_dataset, 
                    batch_size=optimal_batch_size,
                    shuffle=True, 
                    num_workers=min(2, os.cpu_count() or 1),
                    pin_memory=True
                )
            
            # Train on regular games
            train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, 
                    epochs=5, processed_games=processed_games, device=device)
            
            # Clean up
            del regular_games
            del game_dataset
            del game_dataloader
            gc.collect()
            if HAS_OPTIMIZATIONS:
                aggressive_memory_cleanup()
                print_memory_stats("after cleanup")
            else:
                clear_memory()

            print("Running enhanced tactical training phase for regular games...")
            tactical_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            train_tactical(model, tactical_optimizer, puzzle_dataloader, device, epochs=6)
            
            # Generate some self-play games after regular batch
            self_play_count = min(5 + iterations // 2, 10)
            print(f"Generating {self_play_count} self-play games...")
            # Use MCTS by default now
            self_play_games = generate_self_play_games(model, device, num_games=self_play_count, use_mcts=use_mcts)
            
            if self_play_games:
                self_play_dataset = ChessDataset(self_play_games, augment=True)
                self_play_dataloader = DataLoader(
                    self_play_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True
                )
                train_batch(model, self_play_dataloader, puzzle_dataloader, save_path, state_file, 
                        epochs=1, processed_games=processed_games, device=device)
                
                del self_play_games
                del self_play_dataset
                del self_play_dataloader
                gc.collect()
                clear_memory()

            
            # Run tactical test occasionally
            if iterations % 3 == 0:
                test_accuracy = test_tactical_recognition(model, device)
                print(f"Tactical recognition accuracy: {test_accuracy:.2%}")
            
            # Only switch modes if not locked
            if not is_mode_locked and iterations % 3 == 0 and not pro_state.get("all_pro_games_processed", False):
                current_phase = "pro"
                print("Switching to professional games mode for variety")
        
        # Save checkpoint every iteration
        torch.save(model.state_dict(), save_path)
        print(f"Saved model checkpoint (iteration {iterations})")
        
        # Update tracking
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                processed_games = state.get("processed_games", 0)
        
        if os.path.exists(pro_state_file):
            with open(pro_state_file, 'r') as f:
                pro_state = json.load(f)
                pro_game_count = pro_state.get("processed_pro_games", 0)
        
        print(f"Progress: {pro_game_count} professional games, {processed_games} regular games")
        
        # Give user a chance to interrupt gracefully
        print("Waiting 5 seconds before next iteration (Ctrl+C to stop)...")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None, model, save_path, state_file, processed_games, current_epoch)
            break
    
    print(f"\nTraining completed with {processed_games} regular games and {pro_game_count} professional games!")


if __name__ == "__main__":
    main()