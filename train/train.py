import time
import chess
import chess.pgn
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import glob
import json
import signal
import sys
import gc
import random

# Import from our refactored modules
from models import ChessNet
from data import (ChessDataset, PuzzleDataset, load_puzzles, load_lichess_puzzles, 
                 filter_and_prioritize_puzzles_cached, load_professional_games, 
                 load_games_in_batches)
from utils import clear_memory, test_tactical_recognition
from self_play import generate_self_play_games, run_self_play_training
from training import train_batch, train_tactical, get_optimal_batch_size

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Signal handler 
def signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch):
    print("\nTraining interrupted! Saving model and state...")
    torch.save(model.state_dict(), save_path)
    with open(state_file, 'w') as f:
        state = {"processed_games": processed_games, "last_epoch": current_epoch + 1}
        json.dump(state, f)
    print(f"Model saved to {save_path}, state saved to {state_file}")
    sys.exit(0)

# Main execution function
def main():
    # Initialize model and basic setup
    model = ChessNet(num_blocks=10, channels=256).to(device)  
    save_path = "./chess_model/chess_model.pth"
    state_file = "./chess_model/training_state.json"
    pro_state_file = "./chess_model/pro_training_state.json"
    
    # Create directory if it doesn't exist
    os.makedirs("./chess_model", exist_ok=True)
    
    if os.path.exists(save_path):
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
    puzzle_dataset = PuzzleDataset(prioritized_puzzles)
    print(f"Total puzzles after prioritization: {len(prioritized_puzzles)}")

    # Find optimal batch size
    print("Determining optimal batch size...")
    model.eval()
    optimal_batch_size = get_optimal_batch_size(model, device, starting_size=64)
    model.train()
    print(f"Using optimal batch size: {optimal_batch_size}")
    
    # Create puzzle dataloader once - puzzles are smaller and reused
    puzzle_dataloader = DataLoader(
        puzzle_dataset, 
        batch_size=min(32, optimal_batch_size),
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Get current training phase from state file or determine by counts
    if pro_game_count < 1000000:  # Arbitrary threshold for switching to regular games
        current_phase = "pro"
    else:
        current_phase = "regular"
    
    current_phase = "regular"  # Default mode
    if len(sys.argv) > 1:
        if sys.argv[1] in ["pro", "regular", "self-play"]:
            current_phase = sys.argv[1]
            print(f"Command-line override: Using {current_phase} training mode")
    
    # Set batch sizes - smaller batch sizes for faster processing
    pro_batch_size = 1000
    regular_batch_size = 1000
    
    # Main training loop
    max_iterations = 1000  # Safety limit
    iterations = 0
    
    while iterations < max_iterations:
        iterations += 1
        print(f"\n--- Training Iteration {iterations} ---")
        
        if current_phase == "self-play":
            print("\n=== SELF-PLAY REINFORCEMENT LEARNING MODE ===")
            num_games = 500  # Default number of self-play games per iteration
            num_iterations = 10  # Default number of iterations
            
            # Check if user specified number of games and iterations
            if len(sys.argv) > 2:
                try:
                    num_games = int(sys.argv[2])
                except ValueError:
                    print(f"Invalid number of games: {sys.argv[2]}. Using default: {num_games}")
                    
            if len(sys.argv) > 3:
                try:
                    num_iterations = int(sys.argv[3])
                except ValueError:
                    print(f"Invalid number of iterations: {sys.argv[3]}. Using default: {num_iterations}")
            
            # Run self-play training
            model = run_self_play_training(
                model, 
                device,
                save_path, 
                state_file, 
                num_games=num_games,
                num_iterations=num_iterations
            )
            
            print("\nSelf-play training completed!")
            sys.exit(0)

        elif current_phase == "pro":
            print("\n=== PROFESSIONAL GAMES TRAINING ===")
            
            # Load one batch of professional games
            pro_games = load_professional_games(pro_state_file, batch_size=pro_batch_size)
            
            if not pro_games:
                print("No more professional games to process. Permanently switching to regular games.")
                # Mark in the state file that we've processed all pro games
                with open(pro_state_file, 'w') as f:
                    pro_state = {"processed_pro_games": pro_game_count,
                                "all_pro_games_processed": True}
                    json.dump(pro_state, f)
                current_phase = "regular"
                continue

            batch_size = len(pro_games)
            print(f"Processing professional batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(pro_games, augment=True)
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
            clear_memory()
            
            # Tactical training after professional batch
            print("Running quick tactical training phase...")
            tactical_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
            train_tactical(model, tactical_optimizer, puzzle_dataloader, device, epochs=3)
            
            # Generate a few self-play games after each pro batch
            self_play_count = min(iterations, 5)
            print(f"Generating {self_play_count} self-play games...")
            self_play_games = generate_self_play_games(model, device, num_games=self_play_count)
            
            if self_play_games:
                self_play_dataset = ChessDataset(self_play_games, augment=True)
                self_play_dataloader = DataLoader(
                    self_play_dataset,
                    batch_size=optimal_batch_size,
                    shuffle=True,
                    num_workers=1,
                    pin_memory=True
                )
                train_batch(model, self_play_dataloader, puzzle_dataloader, save_path, pro_state_file, 
                        epochs=1, processed_games=processed_games, device=device)
                
                del self_play_games
                del self_play_dataset
                del self_play_dataloader
                gc.collect()
                clear_memory()
            
            # Determine if we should move to regular phase (this can be tuned)
            if pro_game_count > 10000 or iterations % 5 == 0:
                current_phase = "regular"
                
        else:  # Regular games phase
            print("\n=== REGULAR GAMES TRAINING ===")
            
            # Load one batch of regular games
            regular_games = load_games_in_batches(pgn_files, state_file, batch_size=regular_batch_size)
            
            if not regular_games:
                print("No regular games available or error loading games.")
                current_phase = "professional"  # Switch back to pro games if issues with regular games
                continue
                
            batch_size = len(regular_games)
            print(f"Processing regular batch with {batch_size} games")
            
            # Create dataset and dataloader for this batch only
            game_dataset = ChessDataset(regular_games, augment=True)
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
            clear_memory()
            
            # Generate some self-play games after regular batch
            self_play_count = min(5 + iterations // 2, 10)
            print(f"Generating {self_play_count} self-play games...")
            self_play_games = generate_self_play_games(model, device, num_games=self_play_count)
            
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
            
            # Switch back to pro games every few iterations
            if iterations % 3 == 0 and not pro_state.get("all_pro_games_processed", False):
                current_phase = "professional"
        
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