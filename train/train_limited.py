"""
Overnight Training Script for Low VRAM GPUs (GTX 1050 2GB)
==========================================================

This script is optimized for:
- 2GB VRAM (GTX 1050, etc.)
- 8GB System RAM
- Overnight/multi-session training with auto-resume

Features:
- Gradient accumulation (simulates larger batch sizes)
- Automatic checkpointing every epoch
- Auto-resume from last checkpoint
- Memory-efficient data loading
- Progress tracking and ETA
- Graceful interruption handling (Ctrl+C saves checkpoint)

Usage:
    python train_limited.py                    # Start/resume tactical training
    python train_limited.py --mode selfplay    # Self-play training (after tactical)
    python train_limited.py --reset            # Start fresh (delete checkpoints)
    python train_limited.py --status           # Show training progress
"""

import os
import sys
import time
import signal
import argparse
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import chess
import chess.pgn

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import create_chess_model, load_model_with_compatibility
from data import board_to_tensor, ChessDataset, load_games_from_folder
from mcts import select_move_with_mcts, generate_mcts_game

# ============================================================================
# CONFIGURATION FOR LIMITED HARDWARE
# ============================================================================
LIMITED_CONFIG = {
    # Model settings
    "model_type": "limited",
    
    # Training hyperparameters (optimized for 2GB VRAM)
    "batch_size": 16,              # Small batch to fit in VRAM
    "gradient_accumulation": 4,    # Effective batch = 16 * 4 = 64
    "learning_rate": 0.01,         # Lower LR for stability with small batches
    "momentum": 0.9,
    "weight_decay": 1e-4,
    
    # Training schedule
    "tactical_epochs": 100,        # Phase 1: Tactical training
    "selfplay_epochs": 500,        # Phase 2: Self-play (ongoing)
    "selfplay_games_per_epoch": 10,  # Games per self-play epoch
    
    # MCTS settings (reduced for speed)
    "mcts_simulations": 50,        # Fewer sims for faster games
    "mcts_cpuct": 2.0,
    
    # Checkpointing
    "checkpoint_dir": "checkpoints_limited",
    "checkpoint_every": 1,         # Save every epoch
    "keep_last_n": 5,              # Keep last 5 checkpoints
    
    # Memory optimization
    "num_workers": 0,              # No multiprocessing (saves RAM)
    "pin_memory": False,           # Disable for low RAM
    "prefetch_factor": None,
    
    # Data paths
    "pgn_folder": "chess_pgns",
    "puzzle_cache": "cache/puzzle_cache_limited.pkl",
}


# ============================================================================
# TRAINING STATE MANAGEMENT
# ============================================================================
class TrainingState:
    """Manages training state for pause/resume functionality."""
    
    def __init__(self, config):
        self.config = config
        self.checkpoint_dir = Path(config["checkpoint_dir"])
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.state_file = self.checkpoint_dir / "training_state.json"
        
        # Training state
        self.current_phase = "tactical"  # "tactical" or "selfplay"
        self.current_epoch = 0
        self.total_epochs_trained = 0
        self.best_loss = float('inf')
        self.training_history = []
        self.start_time = None
        self.total_training_time = 0
        
        # Load existing state if available
        self.load_state()
    
    def load_state(self):
        """Load training state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.current_phase = state.get("current_phase", "tactical")
                self.current_epoch = state.get("current_epoch", 0)
                self.total_epochs_trained = state.get("total_epochs_trained", 0)
                self.best_loss = state.get("best_loss", float('inf'))
                self.training_history = state.get("training_history", [])
                self.total_training_time = state.get("total_training_time", 0)
                print(f"✓ Resumed from: Phase={self.current_phase}, Epoch={self.current_epoch}")
            except Exception as e:
                print(f"Warning: Could not load state: {e}")
    
    def save_state(self):
        """Save training state to disk."""
        state = {
            "current_phase": self.current_phase,
            "current_epoch": self.current_epoch,
            "total_epochs_trained": self.total_epochs_trained,
            "best_loss": self.best_loss,
            "training_history": self.training_history[-100:],  # Keep last 100
            "total_training_time": self.total_training_time,
            "last_saved": datetime.datetime.now().isoformat(),
        }
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
    
    def get_latest_checkpoint(self):
        """Get path to latest model checkpoint."""
        checkpoints = sorted(self.checkpoint_dir.glob("model_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        return None
    
    def save_checkpoint(self, model, optimizer, scheduler, loss):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"model_epoch_{self.current_epoch:04d}.pt"
        
        torch.save({
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'config': self.config,
        }, checkpoint_path)
        
        # Also save as best if it's the best loss
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.checkpoint_dir / "model_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  ★ New best model! Loss: {loss:.4f}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def _cleanup_old_checkpoints(self):
        """Keep only the last N checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("model_epoch_*.pt"))
        keep_n = self.config.get("keep_last_n", 5)
        
        for old_ckpt in checkpoints[:-keep_n]:
            old_ckpt.unlink()
    
    def load_checkpoint(self, model, optimizer=None, scheduler=None):
        """Load latest checkpoint into model."""
        checkpoint_path = self.get_latest_checkpoint()
        
        if checkpoint_path is None:
            print("No checkpoint found. Starting fresh.")
            return False
        
        print(f"Loading checkpoint: {checkpoint_path.name}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return True


# ============================================================================
# MEMORY-EFFICIENT TACTICAL DATA LOADING
# ============================================================================
def load_tactical_data_limited(config):
    """Load tactical training data with memory constraints."""
    from data import create_tactical_dataset
    
    print("\n📁 Loading tactical training data...")
    
    # Load PGN games
    pgn_folder = Path(config["pgn_folder"])
    if not pgn_folder.exists():
        pgn_folder = Path(__file__).parent / config["pgn_folder"]
    
    if pgn_folder.exists():
        games = load_games_from_folder(str(pgn_folder))
        print(f"  Loaded {len(games)} games from PGNs")
    else:
        games = []
        print(f"  Warning: PGN folder not found at {pgn_folder}")
    
    # Create dataset with limited model type
    dataset = ChessDataset(games, augment=True, model_type="small")  # 18 channels
    
    # Create data loader optimized for low memory
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        drop_last=True,
    )
    
    print(f"  Dataset size: {len(dataset)} positions")
    print(f"  Batches per epoch: {len(dataloader)}")
    
    return dataloader


# ============================================================================
# TRAINING LOOP WITH GRADIENT ACCUMULATION
# ============================================================================
def train_epoch_limited(model, dataloader, optimizer, scheduler, config, device):
    """Train one epoch with gradient accumulation for low VRAM."""
    model.train()
    
    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    num_batches = 0
    
    accumulation_steps = config["gradient_accumulation"]
    optimizer.zero_grad()
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    for batch_idx, batch in enumerate(dataloader):
        positions = batch['position'].to(device)
        policy_targets = batch['policy'].to(device)
        value_targets = batch['value'].to(device)
        
        # Forward pass
        policy_out, value_out = model(positions)
        
        # Calculate losses
        policy_loss = policy_criterion(policy_out, policy_targets)
        value_loss = value_criterion(value_out.squeeze(), value_targets)
        loss = policy_loss + value_loss
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
        
        # Track losses (unscaled)
        total_loss += loss.item() * accumulation_steps
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        num_batches += 1
        
        # Free memory
        del positions, policy_targets, value_targets, policy_out, value_out, loss
        
        # Progress update every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {total_loss/num_batches:.4f}")
    
    # Handle remaining gradients
    if num_batches % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Step scheduler
    if scheduler:
        scheduler.step()
    
    # Return average losses
    return {
        'loss': total_loss / num_batches,
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches,
    }


# ============================================================================
# SELF-PLAY TRAINING FOR LIMITED MODEL
# ============================================================================
def generate_selfplay_game_limited(model, config, device):
    """Generate a single self-play game with limited resources."""
    model.eval()
    
    board = chess.Board()
    game_data = []
    move_count = 0
    max_moves = 150
    
    with torch.no_grad():
        while not board.is_game_over() and move_count < max_moves:
            # Get move from MCTS
            move, policy_probs = select_move_with_mcts(
                board, model, device,
                num_simulations=config["mcts_simulations"],
                temperature=1.0 if move_count < 20 else 0.1
            )
            
            if move is None:
                break
            
            # Store position data
            tensor = board_to_tensor(board, input_channels=18)
            game_data.append({
                'position': tensor,
                'policy': policy_probs,
                'turn': board.turn,
            })
            
            # Make move
            board.push(move)
            move_count += 1
    
    # Determine game result
    if board.is_checkmate():
        result = -1 if board.turn == chess.WHITE else 1
    else:
        result = 0
    
    # Assign values based on result and perspective
    for i, data in enumerate(game_data):
        turn = data['turn']
        data['value'] = result if turn == chess.WHITE else -result
    
    return game_data


def train_selfplay_epoch_limited(model, optimizer, config, device, state):
    """Run one epoch of self-play training."""
    print(f"\n🎮 Self-play epoch {state.current_epoch}")
    
    all_positions = []
    all_policies = []
    all_values = []
    
    # Generate self-play games
    num_games = config["selfplay_games_per_epoch"]
    for game_idx in range(num_games):
        print(f"  Generating game {game_idx+1}/{num_games}...", end=" ", flush=True)
        
        game_data = generate_selfplay_game_limited(model, config, device)
        
        for data in game_data:
            all_positions.append(data['position'])
            all_policies.append(data['policy'])
            all_values.append(data['value'])
        
        print(f"({len(game_data)} moves)")
    
    if len(all_positions) == 0:
        print("  Warning: No positions generated!")
        return {'loss': 0, 'policy_loss': 0, 'value_loss': 0}
    
    # Train on collected data
    print(f"  Training on {len(all_positions)} positions...")
    
    model.train()
    
    positions = torch.stack(all_positions).to(device)
    # Convert policy probs to tensors
    policies = torch.zeros(len(all_policies), 4672)
    for i, p in enumerate(all_policies):
        if isinstance(p, dict):
            for move_idx, prob in p.items():
                if move_idx < 4672:
                    policies[i, move_idx] = prob
        elif isinstance(p, torch.Tensor):
            policies[i] = p
    policies = policies.to(device)
    values = torch.tensor(all_values, dtype=torch.float32).to(device)
    
    # Training loop with gradient accumulation
    batch_size = config["batch_size"]
    accumulation_steps = config["gradient_accumulation"]
    
    total_loss = 0.0
    num_updates = 0
    optimizer.zero_grad()
    
    indices = torch.randperm(len(positions))
    
    for i in range(0, len(positions), batch_size):
        batch_idx = indices[i:i+batch_size]
        
        batch_pos = positions[batch_idx]
        batch_pol = policies[batch_idx]
        batch_val = values[batch_idx]
        
        policy_out, value_out = model(batch_pos)
        
        # Soft policy loss (KL divergence)
        policy_log = nn.functional.log_softmax(policy_out, dim=1)
        policy_loss = -torch.sum(batch_pol * policy_log) / batch_pol.size(0)
        
        value_loss = nn.functional.mse_loss(value_out.squeeze(), batch_val)
        
        loss = policy_loss + value_loss
        loss = loss / accumulation_steps
        loss.backward()
        
        if (num_updates + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_updates += 1
    
    # Final optimizer step
    if num_updates % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    # Cleanup
    del positions, policies, values
    torch.cuda.empty_cache()
    
    return {
        'loss': total_loss / max(num_updates, 1),
        'policy_loss': 0,
        'value_loss': 0,
        'num_games': num_games,
        'num_positions': len(all_positions),
    }


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Limited Hardware Chess Training")
    parser.add_argument("--mode", choices=["tactical", "selfplay"], default=None,
                        help="Training mode (default: auto-detect from state)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset training (delete checkpoints)")
    parser.add_argument("--status", action="store_true",
                        help="Show training status and exit")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    args = parser.parse_args()
    
    config = LIMITED_CONFIG.copy()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print("🎮 LIMITED HARDWARE CHESS TRAINER")
    print(f"{'='*60}")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")
    
    # Initialize state
    state = TrainingState(config)
    
    # Handle --status flag
    if args.status:
        print(f"\n📊 Training Status:")
        print(f"  Phase: {state.current_phase}")
        print(f"  Current epoch: {state.current_epoch}")
        print(f"  Total epochs trained: {state.total_epochs_trained}")
        print(f"  Best loss: {state.best_loss:.4f}")
        print(f"  Total training time: {state.total_training_time/3600:.1f} hours")
        checkpoint = state.get_latest_checkpoint()
        if checkpoint:
            print(f"  Latest checkpoint: {checkpoint.name}")
        return
    
    # Handle --reset flag
    if args.reset:
        import shutil
        if state.checkpoint_dir.exists():
            shutil.rmtree(state.checkpoint_dir)
            print("✓ Deleted all checkpoints")
        state = TrainingState(config)
    
    # Override mode if specified
    if args.mode:
        state.current_phase = args.mode
        if args.mode == "selfplay" and state.current_epoch == 0:
            # Load best tactical model for self-play
            best_model = state.checkpoint_dir / "model_best.pt"
            if best_model.exists():
                print("Will load best tactical model for self-play")
    
    # Create model
    print(f"\n📦 Creating model...")
    model = create_chess_model("limited")
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Create optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"]
    )
    
    # Create scheduler
    if state.current_phase == "tactical":
        total_epochs = args.epochs or config["tactical_epochs"]
    else:
        total_epochs = args.epochs or config["selfplay_epochs"]
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs, eta_min=1e-5
    )
    
    # Load checkpoint if available
    state.load_checkpoint(model, optimizer, scheduler)
    
    # Setup graceful interruption
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        if not interrupted:
            print("\n\n⚠️  Interrupt received! Saving checkpoint...")
            interrupted = True
        else:
            print("Force exit.")
            sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Training loop
    print(f"\n🚀 Starting {state.current_phase} training...")
    print(f"  Starting from epoch {state.current_epoch}")
    print(f"  Target epochs: {total_epochs}")
    print(f"  Batch size: {config['batch_size']} x {config['gradient_accumulation']} = "
          f"{config['batch_size'] * config['gradient_accumulation']} effective")
    
    # Load data for tactical training
    if state.current_phase == "tactical":
        dataloader = load_tactical_data_limited(config)
    
    state.start_time = time.time()
    
    try:
        while state.current_epoch < total_epochs and not interrupted:
            epoch_start = time.time()
            
            print(f"\n{'─'*50}")
            print(f"Epoch {state.current_epoch + 1}/{total_epochs}")
            print(f"{'─'*50}")
            
            # Train one epoch
            if state.current_phase == "tactical":
                metrics = train_epoch_limited(
                    model, dataloader, optimizer, scheduler, config, device
                )
            else:
                metrics = train_selfplay_epoch_limited(
                    model, optimizer, config, device, state
                )
            
            epoch_time = time.time() - epoch_start
            state.total_training_time += epoch_time
            
            # Log metrics
            print(f"\n  📈 Epoch {state.current_epoch + 1} Results:")
            print(f"     Loss: {metrics['loss']:.4f}")
            print(f"     Time: {epoch_time:.1f}s")
            print(f"     LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Update state
            state.current_epoch += 1
            state.total_epochs_trained += 1
            state.training_history.append({
                'epoch': state.current_epoch,
                'phase': state.current_phase,
                'loss': metrics['loss'],
                'time': epoch_time,
            })
            
            # Save checkpoint
            state.save_checkpoint(model, optimizer, scheduler, metrics['loss'])
            state.save_state()
            
            # Estimate remaining time
            avg_epoch_time = state.total_training_time / state.total_epochs_trained
            remaining_epochs = total_epochs - state.current_epoch
            eta = avg_epoch_time * remaining_epochs
            print(f"     ETA: {eta/60:.1f} minutes ({eta/3600:.1f} hours)")
            
            # Memory cleanup
            torch.cuda.empty_cache()
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Always save on exit
        print("\n💾 Saving final checkpoint...")
        if state.current_epoch > 0:
            state.save_checkpoint(model, optimizer, scheduler, 
                                  state.training_history[-1]['loss'] if state.training_history else 0)
        state.save_state()
        
        # Final summary
        print(f"\n{'='*60}")
        print("📊 Training Summary")
        print(f"{'='*60}")
        print(f"  Phase: {state.current_phase}")
        print(f"  Epochs completed: {state.current_epoch}/{total_epochs}")
        print(f"  Total training time: {state.total_training_time/3600:.2f} hours")
        print(f"  Best loss: {state.best_loss:.4f}")
        print(f"\n  To resume: python train_limited.py")
        print(f"  To check status: python train_limited.py --status")
        
        if state.current_phase == "tactical" and state.current_epoch >= config["tactical_epochs"]:
            print(f"\n  ✅ Tactical training complete!")
            print(f"  To start self-play: python train_limited.py --mode selfplay")


if __name__ == "__main__":
    main()
