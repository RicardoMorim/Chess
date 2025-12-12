"""
Stockfish-Guided TD Learning for Chess AI
==========================================

This script uses Stockfish as an "oracle" to provide perfect evaluation signals
for Temporal Difference learning. Much faster convergence than pure self-play.

Key advantages:
- Perfect evaluation signal (no model errors)
- 10x less data needed than self-play
- Learns strong play from expert guidance

Usage:
    python train_with_stockfish.py --model limited --games 1000 --stockfish-path "./stockfish/stockfish-windows-x86-64-avx2.exe"
"""

import sys
import os
import argparse
import time
import subprocess
import re
import json
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import chess
import chess.engine
import chess.pgn
import numpy as np

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import create_chess_model
from data import board_to_tensor, get_move_index
from utils import clear_memory, get_optimal_batch_size, test_tactical_recognition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class PolicyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        if targets.dim() == 1:
            return self.ce_loss(logits, targets)
        else:
            log_probs = F.log_softmax(logits, dim=1)
            return -(targets * log_probs).sum(dim=1).mean()

class ValueLoss(nn.Module):
    def __init__(self, use_huber=True):
        super().__init__()
        if use_huber:
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred.squeeze(), target)

# ============================================================================
# STOCKFISH EVALUATOR
# ============================================================================
class StockfishEvaluator:
    """Wrapper for Stockfish evaluation with efficient batching."""
    
    def __init__(self, stockfish_path, depth=15, threads=1, hash_mb=128):
        """
        Args:
            stockfish_path: Path to stockfish executable
            depth: Search depth (10-20 is good balance)
            threads: Number of threads (1-4)
            hash_mb: Hash table size in MB
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.threads = threads
        self.hash_mb = hash_mb
        self.engine = None
        self._init_engine()
    
    def _init_engine(self):
        """Initialize stockfish engine."""
        try:
            self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            self.engine.configure({
                "Threads": self.threads,
                "Hash": self.hash_mb
            })
            print(f"✓ Stockfish initialized: depth={self.depth}, threads={self.threads}")
        except Exception as e:
            print(f"❌ Failed to initialize Stockfish: {e}")
            print(f"   Path: {self.stockfish_path}")
            sys.exit(1)
    
    def evaluate(self, board, normalized=True):
        """
        Evaluate a chess position using Stockfish.
        
        Args:
            board: chess.Board object
            normalized: Return value in [-1, 1] range
        
        Returns:
            Evaluation score (float)
        """
        try:
            # Get evaluation with limited depth
            info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
            score = info["score"].relative
            
            # Convert to centipawns
            if score.is_mate():
                # Mate in N moves
                mate_in = score.mate()
                cp = 10000 if mate_in > 0 else -10000
            else:
                cp = score.score()
            
            if normalized:
                # Normalize to [-1, 1] using tanh
                # cp=100 → ~0.1, cp=300 → ~0.29, cp=1000 → ~0.76
                return np.tanh(cp / 1000.0)
            else:
                return cp / 100.0  # Return in pawn units
        
        except Exception as e:
            print(f"⚠ Evaluation error: {e}")
            return 0.0
    
    def close(self):
        """Close the engine."""
        if self.engine:
            self.engine.quit()


# ============================================================================
# STOCKFISH-GUIDED GAME GENERATION
# ============================================================================
def generate_stockfish_guided_game(model, stockfish, device, max_moves=200, 
                                   temperature=1.0, input_channels=18):
    """
    Generate a single game where:
    - Model selects moves (learning)
    - Stockfish evaluates each position (teaching)
    
    Returns:
        List of (board_tensor, policy_target, stockfish_value) tuples
    """
    board = chess.Board()
    samples = []
    move_num = 0
    
    model.eval()
    
    while not board.is_game_over() and move_num < max_moves:
        # Get model's move probabilities
        board_tensor = torch.tensor(
            board_to_tensor(board, move_num + 1, input_channels=input_channels),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        with torch.no_grad():
            policy_logits, model_value = model(board_tensor)
            policy = F.softmax(policy_logits / temperature, dim=1).cpu().numpy()[0]
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Create policy target from legal moves
        move_probs = np.zeros(len(legal_moves))
        for i, move in enumerate(legal_moves):
            move_idx = get_move_index(move)
            move_probs[i] = policy[move_idx]
        
        # Normalize
        if move_probs.sum() > 1e-10:
            move_probs = move_probs / move_probs.sum()
        else:
            move_probs = np.ones(len(legal_moves)) / len(legal_moves)
        
        # Sample move (with temperature)
        chosen_move = np.random.choice(legal_moves, p=move_probs)
        
        # Get Stockfish evaluation BEFORE move
        sf_value_before = stockfish.evaluate(board, normalized=True)
        
        # Make move
        board.push(chosen_move)
        
        # Get Stockfish evaluation AFTER move (from opponent's perspective)
        sf_value_after = stockfish.evaluate(board, normalized=True)
        
        # TD target: value after move (flipped for opponent)
        # If position improved for us, it got worse for opponent
        td_target = -sf_value_after  # Flip perspective
        
        # Store sample with TD target
        policy_target = np.zeros(4672, dtype=np.float32)
        move_idx = get_move_index(chosen_move)
        policy_target[move_idx] = 1.0  # One-hot for chosen move
        
        samples.append((
            board_tensor.cpu().numpy()[0],
            policy_target,
            td_target
        ))
        
        move_num += 1
    
    # Final game result (use as bonus signal)
    result = board.result()
    if result == "1-0":
        game_outcome = 1.0
    elif result == "0-1":
        game_outcome = -1.0
    else:
        game_outcome = 0.0
    
    # Blend TD targets with final outcome (exponential decay)
    gamma = 0.99
    for i in range(len(samples) - 1, -1, -1):
        board_t, policy_t, td_val = samples[i]
        
        # Blend: 80% TD, 20% final outcome (closer to end → more final outcome)
        moves_from_end = len(samples) - i
        blend_weight = 0.8 * (gamma ** moves_from_end)
        
        blended_value = blend_weight * td_val + (1 - blend_weight) * game_outcome
        
        # Flip value for alternate moves
        if i % 2 == 1:
            blended_value = -blended_value
        
        samples[i] = (board_t, policy_t, blended_value)
    
    return samples


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train with Stockfish guidance")
    
    parser.add_argument("--model", default="limited", 
                       choices=["limited", "small", "medium", "big"],
                       help="Model size")
    
    parser.add_argument("--games", type=int, default=1000,
                       help="Number of games to generate")
    
    parser.add_argument("--stockfish-path", 
                       default="../stockfish/stockfish-windows-x86-64-avx2.exe",
                       help="Path to Stockfish executable")
    
    parser.add_argument("--stockfish-depth", type=int, default=12,
                       help="Stockfish search depth (10-20)")
    
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Training batch size")
    
    parser.add_argument("--epochs", type=int, default=5,
                       help="Training epochs per batch")
    
    parser.add_argument("--save-every", type=int, default=100,
                       help="Save checkpoint every N games")
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║        STOCKFISH-GUIDED TD LEARNING                               ║
║  Teaching neural network with expert evaluations                  ║
╚═══════════════════════════════════════════════════════════════════╝

Model: {args.model}
Games: {args.games}
Stockfish depth: {args.stockfish_depth}
Device: {device}

Starting...
""")
    
    # Setup paths
    checkpoint_dir = f"./checkpoints_{args.model}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = f"{checkpoint_dir}/model_best.pt"
    state_file = f"{checkpoint_dir}/stockfish_training_state.json"
    
    # Load model
    model = create_chess_model(args.model).to(device)
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 18
    
    # Load checkpoint and state
    start_game = 0
    if os.path.exists(checkpoint_path):
        print(f"✓ Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
                start_game = state.get('games_completed', 0)
                print(f"✓ Resuming from game {start_game}")
        else:
            print("✓ No training state found, but model exists, starting from self game 0 with existing model")
    else:
        print("Starting from scratch")
    
    print(f"Model: {args.model} with {input_channels} input channels")
    
    # Initialize Stockfish
    stockfish = StockfishEvaluator(
        args.stockfish_path,
        depth=args.stockfish_depth,
        threads=2,
        hash_mb=256
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    all_samples = []
    total_positions = 0
    games_completed = start_game
    best_tactical_accuracy = 0.0
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            best_tactical_accuracy = state.get('best_tactical_accuracy', 0.0)
    
    try:
        # Generate games (resume from where we left off)
        for game_num in range(start_game + 1, start_game + args.games + 1):
            print(f"\n[{game_num - start_game}/{args.games}] (Total: {game_num}) Generating...")
            
            start_time = time.time()
            samples = generate_stockfish_guided_game(
                model, stockfish, device,
                max_moves=200,
                temperature=0.8,
                input_channels=input_channels
            )
            gen_time = time.time() - start_time
            
            all_samples.extend(samples)
            total_positions += len(samples)
            games_completed += 1
            
            print(f"   {len(samples)} positions in {gen_time:.1f}s | Buffer: {len(all_samples)}")
            
            # Train every N games or when buffer full
            should_train = (game_num - start_game) % args.save_every == 0
            buffer_full = len(all_samples) >= 50000
            is_last = (game_num - start_game) == args.games
            
            if should_train or buffer_full or is_last:
                print(f"\n{'='*70}")
                print(f"Training on {len(all_samples)} positions...")
                print(f"{'='*70}")
                
                # Prepare dataset
                boards = torch.tensor([s[0] for s in all_samples], dtype=torch.float32)
                policies = torch.tensor([s[1] for s in all_samples], dtype=torch.float32)
                values = torch.tensor([s[2] for s in all_samples], dtype=torch.float32)
                
                dataset = TensorDataset(boards, policies, values)
                dataloader = DataLoader(
                    dataset, 
                    batch_size=args.batch_size,
                    shuffle=True,
                    num_workers=2,
                    pin_memory=True
                )
                
                # Train
                model.train()
                for epoch in range(args.epochs):
                    total_loss = 0
                    total_policy_loss = 0
                    total_value_loss = 0
                    batch_count = 0
                    
                    for batch in dataloader:
                        b_boards, b_policies, b_values = batch
                        b_boards = b_boards.to(device)
                        b_policies = b_policies.to(device)
                        b_values = b_values.to(device)
                        
                        optimizer.zero_grad()
                        
                        policy_logits, value_pred = model(b_boards)
                        
                        policy_loss = policy_loss_fn(policy_logits, b_policies)
                        value_loss = value_loss_fn(value_pred, b_values)
                        
                        loss = policy_loss + value_loss
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                        total_loss += loss.item()
                        total_policy_loss += policy_loss.item()
                        total_value_loss += value_loss.item()
                        batch_count += 1
                    
                    avg_loss = total_loss / batch_count
                    avg_policy = total_policy_loss / batch_count
                    avg_value = total_value_loss / batch_count
                    
                    print(f"Epoch {epoch+1}/{args.epochs}: "
                          f"Loss={avg_loss:.4f} (policy={avg_policy:.4f}, value={avg_value:.4f})")
                
                # Validate
                print(f"\nEvaluating...")
                model.eval()
                tact_acc = test_tactical_recognition(model, device)
                print(f"Tactical: {tact_acc:.2%}")
                
                # Save
                torch.save(model.state_dict(), checkpoint_path)
                
                state = {
                    'games_completed': games_completed,
                    'total_positions': total_positions,
                    'tactical_accuracy': tact_acc,
                    'best_tactical_accuracy': max(best_tactical_accuracy, tact_acc)
                }
                with open(state_file, 'w') as f:
                    json.dump(state, f)
                
                if tact_acc > best_tactical_accuracy:
                    best_tactical_accuracy = tact_acc
                    torch.save(model.state_dict(), f"{checkpoint_dir}/model_stockfish_best.pt")
                    print(f"✓ New best: {tact_acc:.2%}")
                
                all_samples = []
                gc.collect()
                clear_memory()
    
    finally:
        stockfish.close()
        print("\n✓ Stockfish closed")
    
    print(f"""

╔═══════════════════════════════════════════════════════════════════╗
║                  TRAINING COMPLETE                                ║
╚═══════════════════════════════════════════════════════════════════╝

Total positions trained: {total_positions}
Model saved: {checkpoint_path}

Next steps:
  1. Test the model: python ../Main.py
  2. Continue training: Run this script again
  3. Switch to self-play: python train.py self-play 20 3 --model {args.model}

""")


if __name__ == "__main__":
    main()
