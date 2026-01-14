"""
Advanced Training Utilities for Chess AI
=========================================

This module provides advanced training utilities:
- Full checkpoint saving/loading (optimizer, scheduler, EMA)
- EMA (Exponential Moving Average) for stable model weights
- TensorBoard logging
- Stockfish evaluation for model validation
- Validation during training

Usage:
    from training_utils import CheckpointManager, EMA, TensorBoardLogger, StockfishEvaluator
"""

import os
import copy
import time
import torch
import torch.nn as nn
import chess
import numpy as np
from typing import Dict, Optional, List, Tuple
from collections import OrderedDict


# ============================================================================
# EXPONENTIAL MOVING AVERAGE (EMA)
# ============================================================================
class EMA:
    """Exponential Moving Average of model parameters.
    
    EMA provides more stable model weights by maintaining a running average
    of the model parameters. This often leads to better generalization.
    
    Usage:
        ema = EMA(model, decay=0.999)
        for batch in dataloader:
            loss = train_step(model, batch)
            loss.backward()
            optimizer.step()
            ema.update()  # Update EMA after each step
        
        # For evaluation, use EMA weights
        ema.apply_shadow()  # Apply EMA weights to model
        evaluate(model)
        ema.restore()  # Restore original weights
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """Initialize EMA.
        
        Args:
            model: The model to track
            decay: EMA decay rate (higher = slower updates, more stable)
                   Typical values: 0.999 for large models, 0.99 for small
        """
        self.model = model
        self.decay = decay
        self.shadow = {}  # EMA weights
        self.backup = {}  # Original weights backup
        self.num_updates = 0
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA weights after an optimizer step."""
        self.num_updates += 1
        
        # Use warmup: decay increases from 0 to target over first 2000 steps
        # This prevents early unstable weights from dominating
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # EMA update: shadow = decay * shadow + (1 - decay) * param
                    self.shadow[name].mul_(decay).add_(param.data, alpha=1 - decay)
    
    def apply_shadow(self):
        """Apply EMA weights to the model (backup original weights first)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original weights from backup."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get EMA state for checkpointing."""
        return {
            'shadow': {k: v.cpu() for k, v in self.shadow.items()},
            'num_updates': self.num_updates
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint."""
        self.num_updates = state_dict.get('num_updates', 0)
        for name, param in state_dict.get('shadow', {}).items():
            if name in self.shadow:
                self.shadow[name].copy_(param.to(self.shadow[name].device))


# ============================================================================
# CHECKPOINT MANAGER
# ============================================================================
class CheckpointManager:
    """Manages full training state checkpoints.
    
    Saves and loads complete training state including:
    - Model weights
    - Optimizer state
    - Scheduler state
    - EMA weights
    - Training metrics
    - Epoch number
    
    Usage:
        ckpt_manager = CheckpointManager(save_dir, model, optimizer, scheduler, ema)
        
        # Save checkpoint
        ckpt_manager.save(epoch, metrics={'val_loss': 0.5, 'val_acc': 0.8})
        
        # Load checkpoint
        start_epoch = ckpt_manager.load()
    """
    
    def __init__(self, save_dir: str, model: nn.Module, optimizer=None, 
                 scheduler=None, ema: Optional[EMA] = None, keep_last_n: int = 3):
        """Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints
            model: The model to checkpoint
            optimizer: Optional optimizer to checkpoint
            scheduler: Optional LR scheduler to checkpoint
            ema: Optional EMA instance to checkpoint
            keep_last_n: Number of recent checkpoints to keep (0 = keep all)
        """
        self.save_dir = save_dir
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.ema = ema
        self.keep_last_n = keep_last_n
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.best_metric = None
        self.best_epoch = 0
    
    def save(self, epoch: int, metrics: Dict = None, is_best: bool = False):
        """Save a checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics to save
            is_best: If True, also save as 'model_best.pt'
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': metrics or {},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if self.optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch:04d}.pt')
        torch.save(checkpoint, epoch_path)
        print(f"💾 Saved checkpoint: {epoch_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.save_dir, 'model_best.pt')
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved!")
            self.best_epoch = epoch
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def load(self, checkpoint_path: str = None) -> int:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Specific checkpoint to load, or None for latest
        
        Returns:
            Starting epoch number (0 if no checkpoint found)
        """
        if checkpoint_path is None:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(self.save_dir) 
                          if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
            if not checkpoints:
                print("No checkpoint found, starting from scratch")
                return 0
            checkpoints.sort()
            checkpoint_path = os.path.join(self.save_dir, checkpoints[-1])
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0
        
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer
        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load EMA
        if self.ema and 'ema_state_dict' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        metrics = checkpoint.get('metrics', {})
        print(f"  Resumed from epoch {epoch}, metrics: {metrics}")
        
        return epoch + 1  # Return next epoch to train
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the N most recent."""
        if self.keep_last_n <= 0:
            return
        
        checkpoints = [f for f in os.listdir(self.save_dir) 
                      if f.startswith('checkpoint_epoch_') and f.endswith('.pt')]
        checkpoints.sort()
        
        # Keep best and last N
        to_remove = checkpoints[:-self.keep_last_n]
        for ckpt in to_remove:
            path = os.path.join(self.save_dir, ckpt)
            os.remove(path)


# ============================================================================
# TENSORBOARD LOGGER
# ============================================================================
class TensorBoardLogger:
    """TensorBoard logging for training metrics.
    
    Usage:
        logger = TensorBoardLogger(log_dir='runs/experiment1')
        logger.log_scalar('train/loss', loss, step)
        logger.log_scalars('accuracy', {'train': 0.8, 'val': 0.7}, step)
        logger.close()
    """
    
    def __init__(self, log_dir: str = 'runs'):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        self._init_writer()
    
    def _init_writer(self):
        """Initialize the SummaryWriter."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            print(f"📊 TensorBoard logging to: {self.log_dir}")
            print(f"   Run: tensorboard --logdir={self.log_dir}")
        except ImportError:
            print("⚠️ TensorBoard not installed. Run: pip install tensorboard")
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars under one main tag."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram of values."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_model_params(self, model: nn.Module, step: int):
        """Log model parameter histograms."""
        if self.writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.writer.add_histogram(f'params/{name}', param.data, step)
                    if param.grad is not None:
                        self.writer.add_histogram(f'grads/{name}', param.grad, step)
    
    def log_training_step(self, step: int, loss: float, policy_loss: float, 
                          value_loss: float, lr: float):
        """Log standard training metrics."""
        if self.writer:
            self.writer.add_scalar('train/loss', loss, step)
            self.writer.add_scalar('train/policy_loss', policy_loss, step)
            self.writer.add_scalar('train/value_loss', value_loss, step)
            self.writer.add_scalar('train/learning_rate', lr, step)
    
    def log_validation(self, step: int, val_loss: float, val_accuracy: float,
                       tactical_accuracy: float = None):
        """Log validation metrics."""
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, step)
            self.writer.add_scalar('val/accuracy', val_accuracy, step)
            if tactical_accuracy is not None:
                self.writer.add_scalar('val/tactical_accuracy', tactical_accuracy, step)
    
    def log_stockfish_eval(self, step: int, correlation: float, mae: float):
        """Log Stockfish evaluation metrics."""
        if self.writer:
            self.writer.add_scalar('eval/stockfish_correlation', correlation, step)
            self.writer.add_scalar('eval/stockfish_mae', mae, step)
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


# ============================================================================
# STOCKFISH EVALUATOR
# ============================================================================
class StockfishEvaluator:
    """Evaluate model predictions against Stockfish.
    
    Compares model value predictions with Stockfish evaluations
    to measure how well the model understands position strength.
    
    Usage:
        evaluator = StockfishEvaluator(stockfish_path='stockfish.exe')
        correlation, mae = evaluator.evaluate_model(model, positions, device)
    """
    
    def __init__(self, stockfish_path: str = None, depth: int = 12, time_limit: float = 0.1):
        """Initialize Stockfish evaluator.
        
        Args:
            stockfish_path: Path to Stockfish executable (None for auto-detect)
            depth: Search depth for Stockfish
            time_limit: Time limit per position in seconds
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit
        self.engine = None
        self._init_stockfish()
    
    def _init_stockfish(self):
        """Initialize Stockfish engine."""
        try:
            import chess.engine
            
            # Try to find Stockfish - check project folder first
            paths_to_try = [self.stockfish_path] if self.stockfish_path else []
            
            # Get project root (parent of train folder)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            paths_to_try.extend([
                # Project-relative paths (your chess repository)
                os.path.join(project_root, 'stockfish', 'stockfish.exe'),
                os.path.join(project_root, 'stockfish', 'stockfish-windows-x86-64-avx2.exe'),
                os.path.join(project_root, 'stockfish', 'stockfish'),
                '../stockfish/stockfish.exe',
                '../stockfish/stockfish',
                # Common system paths
                'stockfish',
                'stockfish.exe',
                r'C:\stockfish\stockfish.exe',
                '/usr/local/bin/stockfish',
                '/usr/bin/stockfish',
            ])
            
            for path in paths_to_try:
                if path and os.path.exists(path):
                    try:
                        self.engine = chess.engine.SimpleEngine.popen_uci(path)
                        print(f"♟️ Stockfish loaded from: {path}")
                        return
                    except Exception:
                        continue
            
            print("⚠️ Stockfish not found. Position evaluation disabled.")
            print("   Download from: https://stockfishchess.org/download/")
            
        except ImportError:
            print("⚠️ chess.engine not available")
    
    def get_stockfish_eval(self, board: chess.Board) -> float:
        """Get Stockfish evaluation for a position.
        
        Args:
            board: Chess board position
        
        Returns:
            Evaluation in centipawns, normalized to [-1, 1] range
        """
        if self.engine is None:
            return 0.0
        
        try:
            import chess.engine
            result = self.engine.analyse(
                board, 
                chess.engine.Limit(depth=self.depth, time=self.time_limit)
            )
            score = result['score'].relative
            
            if score.is_mate():
                # Convert mate score to ±1
                mate_in = score.mate()
                return 1.0 if mate_in > 0 else -1.0
            else:
                # Normalize centipawn score to [-1, 1] using tanh
                cp = score.score()
                return np.tanh(cp / 400.0)  # ±400cp → ±0.76
                
        except Exception as e:
            return 0.0
    
    def evaluate_model(self, model: nn.Module, positions: List[chess.Board], 
                       device, input_channels: int = 22) -> Tuple[float, float]:
        """Evaluate model predictions against Stockfish.
        
        Args:
            model: The chess model
            positions: List of chess positions to evaluate
            device: Computation device
            input_channels: Model input channels
        
        Returns:
            Tuple of (correlation, mean absolute error)
        """
        if self.engine is None or not positions:
            return 0.0, 1.0
        
        from data import board_to_tensor
        
        model.eval()
        model_values = []
        stockfish_values = []
        
        print(f"Evaluating {len(positions)} positions against Stockfish...")
        
        with torch.no_grad():
            for board in positions:
                # Get model prediction
                tensor = board_to_tensor(board, 0, input_channels)
                input_tensor = torch.tensor(tensor).unsqueeze(0).to(device)
                _, value = model(input_tensor)
                model_val = value.item()
                
                # Get Stockfish evaluation
                sf_val = self.get_stockfish_eval(board)
                
                # Flip for black to move
                if board.turn == chess.BLACK:
                    model_val = -model_val
                    sf_val = -sf_val
                
                model_values.append(model_val)
                stockfish_values.append(sf_val)
        
        # Calculate metrics
        model_values = np.array(model_values)
        stockfish_values = np.array(stockfish_values)
        
        # Pearson correlation
        if len(model_values) > 1:
            correlation = np.corrcoef(model_values, stockfish_values)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Mean absolute error
        mae = np.mean(np.abs(model_values - stockfish_values))
        
        print(f"  Correlation with Stockfish: {correlation:.3f}")
        print(f"  Mean Absolute Error: {mae:.3f}")
        
        return correlation, mae
    
    def close(self):
        """Close the Stockfish engine."""
        if self.engine:
            self.engine.quit()


# ============================================================================
# VALIDATION UTILITIES
# ============================================================================
def run_validation(model: nn.Module, val_loader, device, 
                   input_channels: int = 22) -> Tuple[float, float, float]:
    """Run validation on a dataset.
    
    Args:
        model: The model to validate
        val_loader: Validation data loader
        device: Computation device
        input_channels: Model input channels
    
    Returns:
        Tuple of (val_loss, policy_accuracy, value_mae)
    """
    from training import PolicyLoss, ValueLoss
    
    model.eval()
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    total_loss = 0
    correct = 0
    total = 0
    value_mae = 0
    batch_count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) >= 3:
                inputs, policy_targets, value_targets = batch[:3]
            else:
                continue
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            policy_logits, value_pred = model(inputs)
            
            # Calculate losses
            p_loss = policy_loss_fn(policy_logits, policy_targets)
            v_loss = value_loss_fn(value_pred, value_targets)
            total_loss += (p_loss.item() + v_loss.item())
            
            # Policy accuracy
            predictions = policy_logits.argmax(dim=1)
            correct += (predictions == policy_targets).sum().item()
            total += policy_targets.size(0)
            
            # Value MAE
            value_mae += torch.abs(value_pred.squeeze() - value_targets).mean().item()
            batch_count += 1
    
    avg_loss = total_loss / max(1, batch_count)
    accuracy = correct / max(1, total)
    avg_value_mae = value_mae / max(1, batch_count)
    
    return avg_loss, accuracy, avg_value_mae


def generate_validation_positions(num_positions: int = 100) -> List[chess.Board]:
    """Generate random positions for Stockfish evaluation.
    
    Creates varied positions by playing random moves from starting position.
    """
    import random
    positions = []
    
    for _ in range(num_positions):
        board = chess.Board()
        
        # Play 5-40 random moves
        num_moves = random.randint(5, 40)
        for _ in range(num_moves):
            legal_moves = list(board.legal_moves)
            if not legal_moves or board.is_game_over():
                break
            board.push(random.choice(legal_moves))
        
        if not board.is_game_over():
            positions.append(board.copy())
    
    return positions
