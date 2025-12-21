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
# STOCKFISH-GUIDED GAME GENERATION (IMPROVED)
# ============================================================================
def generate_stockfish_guided_game(model, stockfish, device, max_moves=200, 
                                   temperature=0.3, input_channels=18, 
                                   exploration_rate=0.1):
    """
    Generate a training game where:
    - Stockfish provides the BEST MOVE as policy target (teaching)
    - Stockfish provides position evaluation as value target
    - Model occasionally plays to explore (with low probability)
    
    Key improvement: Policy target is Stockfish's best move, not model's move!
    
    Args:
        model: Neural network model
        stockfish: StockfishEvaluator instance
        device: Computation device
        max_moves: Maximum moves per game
        temperature: Temperature for model's exploration moves (lower = more greedy)
        input_channels: Number of input channels for the model
        exploration_rate: Probability of using model's move instead of Stockfish's
    
    Returns:
        List of (board_tensor, policy_target, stockfish_value) tuples
    """
    board = chess.Board()
    samples = []
    move_num = 0
    
    model.eval()
    
    while not board.is_game_over() and move_num < max_moves:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Get board tensor for this position (BEFORE making move)
        board_tensor = torch.tensor(
            board_to_tensor(board, move_num + 1, input_channels=input_channels),
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Get Stockfish's evaluation and best move
        try:
            # Get Stockfish analysis with best move
            info = stockfish.engine.analyse(
                board, 
                chess.engine.Limit(depth=stockfish.depth),
                multipv=3  # Get top 3 moves for soft targets
            )
            
            # Handle both single PV and multi-PV results
            if isinstance(info, list):
                # Multi-PV result
                best_info = info[0]
                best_move = best_info.get("pv", [None])[0]
                
                # Create soft policy target from top moves
                policy_target = np.zeros(4672, dtype=np.float32)
                total_weight = 0
                for i, pv_info in enumerate(info):
                    pv_move = pv_info.get("pv", [None])[0]
                    if pv_move and pv_move in legal_moves:
                        # Weight: 1st move gets 0.7, 2nd gets 0.2, 3rd gets 0.1
                        weight = [0.7, 0.2, 0.1][i] if i < 3 else 0.05
                        move_idx = get_move_index(pv_move)
                        policy_target[move_idx] = weight
                        total_weight += weight
                
                # Normalize
                if total_weight > 0:
                    policy_target /= total_weight
                else:
                    # Fallback to uniform over legal moves
                    for move in legal_moves:
                        policy_target[get_move_index(move)] = 1.0 / len(legal_moves)
            else:
                # Single PV result
                best_move = info.get("pv", [None])[0]
                policy_target = np.zeros(4672, dtype=np.float32)
                if best_move:
                    policy_target[get_move_index(best_move)] = 1.0
                else:
                    for move in legal_moves:
                        policy_target[get_move_index(move)] = 1.0 / len(legal_moves)
            
            # Get Stockfish evaluation (value target)
            score = best_info["score"].relative if isinstance(info, list) else info["score"].relative
            if score.is_mate():
                mate_in = score.mate()
                sf_value = 1.0 if mate_in > 0 else -1.0
            else:
                cp = score.score()
                sf_value = np.tanh(cp / 400.0)  # Normalize: 400cp ≈ 0.76
                
        except Exception as e:
            print(f"⚠ Stockfish error: {e}")
            # Fallback: use model's prediction
            with torch.no_grad():
                policy_logits, value_pred = model(board_tensor)
                policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            policy_target = np.zeros(4672, dtype=np.float32)
            for move in legal_moves:
                policy_target[get_move_index(move)] = policy[get_move_index(move)]
            if policy_target.sum() > 0:
                policy_target /= policy_target.sum()
            else:
                for move in legal_moves:
                    policy_target[get_move_index(move)] = 1.0 / len(legal_moves)
            
            best_move = legal_moves[0]
            sf_value = 0.0
        
        # Store the training sample
        samples.append((
            board_tensor.cpu().numpy()[0],
            policy_target,
            sf_value
        ))
        
        # Decide which move to play (mostly Stockfish, sometimes model for exploration)
        if best_move and np.random.random() > exploration_rate:
            # Play Stockfish's best move (90% of the time)
            chosen_move = best_move
        else:
            # Play model's move for exploration (10% of the time)
            with torch.no_grad():
                policy_logits, _ = model(board_tensor)
                policy = F.softmax(policy_logits / temperature, dim=1).cpu().numpy()[0]
            
            move_probs = np.array([policy[get_move_index(m)] for m in legal_moves])
            if move_probs.sum() > 1e-10:
                move_probs /= move_probs.sum()
            else:
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)
            
            chosen_move = np.random.choice(legal_moves, p=move_probs)
        
        # Make the move
        board.push(chosen_move)
        move_num += 1
    
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
    parser.add_argument("--infinite", action="store_true",
                       help="Run indefinitely until stopped (overrides --games)")
    
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
    # Enable infinite mode if games <= 0
    if args.games is not None and args.games <= 0:
        args.infinite = True
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║        STOCKFISH-GUIDED TD LEARNING                               ║
║  Teaching neural network with expert evaluations                  ║
╚═══════════════════════════════════════════════════════════════════╝

Model: {args.model}
Games: {('infinite' if args.infinite else args.games)}
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
    
    # Use SGD for consistency with other training scripts (AlphaZero-style)
    base_lr = 0.005  # Starting LR (will be modulated by scheduler)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    # Loss weights - value head slightly higher for better position understanding
    policy_weight = 1.0
    value_weight = 1.5
    
    # Mixed precision training for speed
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # LR scheduler state
    training_cycles = 0
    warmup_epochs = 2

    all_samples = []
    total_positions = 0
    games_completed = start_game
    best_tactical_accuracy = 0.0
    
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            best_tactical_accuracy = state.get('best_tactical_accuracy', 0.0)
            training_cycles = state.get('training_cycles', 0)
    
    try:
        # Generate games (resume from where we left off)
        if args.infinite:
            print("\nRunning in infinite mode. Press Ctrl+C to stop.")
            game_num = start_game
            while True:
                game_num += 1
                print(f"\n[{games_completed - start_game + 1}/∞] (Total: {game_num}) Generating...")

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
                should_train = (games_completed % args.save_every == 0)
                buffer_full = len(all_samples) >= 50000

                if should_train or buffer_full:
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

                    # Train with improvements
                    model.train()
                    training_cycles += 1
                    
                    # Learning rate schedule: warmup then cosine decay
                    for epoch in range(args.epochs):
                        # Calculate LR with warmup
                        if training_cycles <= warmup_epochs:
                            # Linear warmup
                            current_lr = base_lr * training_cycles / warmup_epochs
                        else:
                            # Cosine decay after warmup
                            progress = (training_cycles - warmup_epochs) / max(1, 100)  # Decay over ~100 cycles
                            current_lr = base_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
                            current_lr = max(current_lr, 1e-5)  # Minimum LR
                        
                        # Update optimizer LR
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        
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

                            # Mixed precision forward pass
                            if scaler:
                                with torch.cuda.amp.autocast():
                                    policy_logits, value_pred = model(b_boards)
                                    policy_loss = policy_loss_fn(policy_logits, b_policies)
                                    value_loss = value_loss_fn(value_pred, b_values)
                                    loss = policy_weight * policy_loss + value_weight * value_loss
                                
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                policy_logits, value_pred = model(b_boards)
                                policy_loss = policy_loss_fn(policy_logits, b_policies)
                                value_loss = value_loss_fn(value_pred, b_values)
                                loss = policy_weight * policy_loss + value_weight * value_loss
                                
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

                        print(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}): "
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
                        'best_tactical_accuracy': max(best_tactical_accuracy, tact_acc),
                        'training_cycles': training_cycles
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
        else:
            # Finite mode
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

                    # Train with improvements
                    model.train()
                    training_cycles += 1
                    
                    # Learning rate schedule: warmup then cosine decay
                    for epoch in range(args.epochs):
                        # Calculate LR with warmup
                        if training_cycles <= warmup_epochs:
                            current_lr = base_lr * training_cycles / warmup_epochs
                        else:
                            progress = (training_cycles - warmup_epochs) / max(1, 100)
                            current_lr = base_lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))
                            current_lr = max(current_lr, 1e-5)
                        
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = current_lr
                        
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

                            if scaler:
                                with torch.cuda.amp.autocast():
                                    policy_logits, value_pred = model(b_boards)
                                    policy_loss = policy_loss_fn(policy_logits, b_policies)
                                    value_loss = value_loss_fn(value_pred, b_values)
                                    loss = policy_weight * policy_loss + value_weight * value_loss
                                
                                scaler.scale(loss).backward()
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                policy_logits, value_pred = model(b_boards)
                                policy_loss = policy_loss_fn(policy_logits, b_policies)
                                value_loss = value_loss_fn(value_pred, b_values)
                                loss = policy_weight * policy_loss + value_weight * value_loss
                                
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

                        print(f"Epoch {epoch+1}/{args.epochs} (LR={current_lr:.6f}): "
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
                        'best_tactical_accuracy': max(best_tactical_accuracy, tact_acc),
                        'training_cycles': training_cycles
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
    except KeyboardInterrupt:
        print("\n⚠ Interrupted. Saving state and checkpoint...")
        # Train on any remaining buffer before exit
        if len(all_samples) > 0:
            boards = torch.tensor([s[0] for s in all_samples], dtype=torch.float32)
            policies = torch.tensor([s[1] for s in all_samples], dtype=torch.float32)
            values = torch.tensor([s[2] for s in all_samples], dtype=torch.float32)

            dataset = TensorDataset(boards, policies, values)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

            model.train()
            for epoch in range(max(1, args.epochs // 2)):
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

        # Save latest
        model.eval()
        torch.save(model.state_dict(), checkpoint_path)
        # Keep previous best if it's better
        state = {
            'games_completed': games_completed,
            'total_positions': total_positions,
            'tactical_accuracy': state.get('tactical_accuracy', 0.0) if 'state' in locals() else 0.0,
            'best_tactical_accuracy': best_tactical_accuracy
        }
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print("✓ State saved. Exiting...")
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
