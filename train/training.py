import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from torch.cuda.amp import autocast, GradScaler
import json
import os
import sys
import time
import math
import gc

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAIN_CONFIG = {
    # Optimizer settings (AlphaZero-style)
    'optimizer': 'sgd',           # 'sgd' or 'adam'
    'sgd_lr': 0.01,                # Initial LR for SGD (reduced from 0.2)
    'sgd_momentum': 0.9,          # Momentum for SGD
    'adam_lr': 0.001,             # Initial LR for Adam
    'weight_decay': 1e-4,         # L2 regularization
    
    # Learning rate schedule
    'lr_schedule': 'cosine',      # 'cosine', 'step', or 'onecycle'
    'lr_milestones': [100, 150],  # For step schedule
    'lr_gamma': 0.1,              # LR decay factor
    
    # Gradient clipping
    'grad_clip': 1.0,             # Max gradient norm
    
    # Loss weights
    'policy_weight': 1.0,         # Policy loss weight
    'value_weight': 1.0,          # Value loss weight
    'puzzle_policy_weight': 2.0,  # Higher weight for puzzles
    'puzzle_value_weight': 2.0,   # Value weight for puzzles (increased)
    
    # Training dynamics
    'puzzle_frequency': 1,        # Train on puzzles every N game batches
    'puzzle_batches': 5,          # Number of puzzle batches per game batch
}


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================
class PolicyLoss(nn.Module):
    """Policy loss that handles both hard targets (indices) and soft targets (distributions)."""
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Policy logits from network (B, num_moves)
            targets: Either class indices (B,) or soft target distributions (B, num_moves)
        """
        if targets.dim() == 1:
            # Hard targets (class indices) - use cross entropy
            return self.ce_loss(logits, targets)
        else:
            # Soft targets (probability distributions) - use KL divergence
            log_probs = F.log_softmax(logits, dim=1)
            return -(targets * log_probs).sum(dim=1).mean()


class ValueLoss(nn.Module):
    """Value loss with optional Huber loss for robustness."""
    def __init__(self, use_huber=False, delta=1.0):
        super().__init__()
        self.use_huber = use_huber
        if use_huber:
            self.loss_fn = nn.SmoothL1Loss(beta=delta)
        else:
            self.loss_fn = nn.MSELoss()
    
    def forward(self, pred, target):
        return self.loss_fn(pred.squeeze(), target)


# ============================================================================
# OPTIMIZER FACTORY
# ============================================================================
def create_optimizer(model, config=None):
    """Create optimizer based on configuration.
    
    AlphaZero uses SGD with momentum, which often works better than Adam
    for self-play reinforcement learning.
    """
    if config is None:
        config = TRAIN_CONFIG
    
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config['sgd_lr'],
            momentum=config['sgd_momentum'],
            weight_decay=config['weight_decay'],
            nesterov=True  # Nesterov momentum often helps
        )
    else:  # adam
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['adam_lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
    
    return optimizer


def create_scheduler(optimizer, num_epochs, config=None, warmup_epochs=5):
    """Create learning rate scheduler with optional warmup.
    
    Args:
        optimizer: The optimizer
        num_epochs: Total training epochs
        config: Training config
        warmup_epochs: Number of warmup epochs (linear ramp)
    """
    if config is None:
        config = TRAIN_CONFIG
    
    # Get warmup from config if specified
    warmup = config.get('lr_warmup_epochs', warmup_epochs)
    
    if config['lr_schedule'] == 'cosine':
        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup:
                # Linear warmup
                return (epoch + 1) / warmup
            else:
                # Cosine annealing after warmup
                progress = (epoch - warmup) / max(1, num_epochs - warmup)
                return max(0.01, 0.5 * (1 + math.cos(math.pi * progress)))
        scheduler = LambdaLR(optimizer, lr_lambda)
    elif config['lr_schedule'] == 'onecycle':
        # OneCycleLR is good for training from scratch
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['sgd_lr'] if config['optimizer'] == 'sgd' else config['adam_lr'],
            epochs=num_epochs,
            steps_per_epoch=1,  # Will be updated
            pct_start=0.3,
            anneal_strategy='cos'
        )
    else:  # step
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config['lr_milestones'],
            gamma=config['lr_gamma']
        )
    
    return scheduler


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, 
                epochs=5, processed_games=0, device='cuda', use_sgd=True):
    """Train the model on a batch of games and puzzles.
    
    Improvements over original:
    - Option to use SGD with momentum (AlphaZero-style)
    - Gradient clipping for stability
    - Better loss weighting
    - Support for soft policy targets
    """
    # Create optimizer
    if use_sgd:
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=TRAIN_CONFIG['sgd_lr'],  # Use config value
            momentum=0.9, 
            weight_decay=1e-4,
            nesterov=True
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Use scheduler with warmup
    scheduler = create_scheduler(optimizer, epochs, TRAIN_CONFIG)
    
    # Loss functions
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)  # Huber loss is more robust
    
    # Gradient scaler for mixed precision
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Configuration
    grad_clip = TRAIN_CONFIG['grad_clip']
    policy_weight = TRAIN_CONFIG['policy_weight']
    value_weight = TRAIN_CONFIG['value_weight']
    puzzle_policy_weight = TRAIN_CONFIG['puzzle_policy_weight']
    puzzle_value_weight = TRAIN_CONFIG['puzzle_value_weight']
    puzzle_frequency = TRAIN_CONFIG['puzzle_frequency']
    puzzle_batches = TRAIN_CONFIG['puzzle_batches']

    # Load state
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            start_epoch = state.get("last_epoch", 0)
            print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        state = {"processed_games": processed_games, "last_epoch": 0}
        start_epoch = 0

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        game_batch_count = 0
        
        # Create fresh puzzle iterator for this epoch to allow garbage collection
        import itertools
        puzzle_iter = itertools.cycle(puzzle_dataloader)
        
        epoch_start = time.time()
        
        for game_batch in game_dataloader:
            inputs, policy_targets, value_targets = game_batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with autocast():
                    policy_logits, value_pred = model(inputs)
                    policy_loss = policy_loss_fn(policy_logits, policy_targets)
                    value_loss = value_loss_fn(value_pred, value_targets)
                    loss = policy_weight * policy_loss + value_weight * value_loss
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                policy_logits, value_pred = model(inputs)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred, value_targets)
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            game_batch_count += 1

            # Interleaved puzzle training
            if game_batch_count % puzzle_frequency == 0:
                for _ in range(puzzle_batches):
                    puzzle_batch = next(puzzle_iter)
                    p_inputs, p_policy_targets, p_value_targets = puzzle_batch[:3]
                    p_inputs = p_inputs.to(device)
                    p_policy_targets = p_policy_targets.to(device)
                    p_value_targets = p_value_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    if scaler:
                        with autocast():
                            p_logits, p_value = model(p_inputs)
                            p_policy_loss = policy_loss_fn(p_logits, p_policy_targets)
                            p_value_loss = value_loss_fn(p_value, p_value_targets)
                            p_loss = puzzle_policy_weight * p_policy_loss + puzzle_value_weight * p_value_loss
                        
                        scaler.scale(p_loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        p_logits, p_value = model(p_inputs)
                        p_policy_loss = policy_loss_fn(p_logits, p_policy_targets)
                        p_value_loss = value_loss_fn(p_value, p_value_targets)
                        p_loss = puzzle_policy_weight * p_policy_loss + puzzle_value_weight * p_value_loss
                        
                        p_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    
                    total_loss += p_loss.item()

        # Update scheduler
        scheduler.step()
        
        del puzzle_iter
        gc.collect()

        # Logging
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(1, game_batch_count * (1 + puzzle_batches))
        avg_policy = total_policy_loss / max(1, game_batch_count)
        avg_value = total_value_loss / max(1, game_batch_count)
        
        print(f"Epoch {epoch + 1}/{epochs + start_epoch}")
        print(f"  Loss: {avg_loss:.4f} (policy: {avg_policy:.4f}, value: {avg_value:.4f})")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}, Time: {epoch_time:.1f}s")

        # Save checkpoint
        state["last_epoch"] = epoch + 1
        state["processed_games"] = processed_games
        
        torch.save(model.state_dict(), save_path)
        with open(state_file, 'w') as f:
            json.dump(state, f)


# ============================================================================
# TACTICAL TRAINING
# ============================================================================
def train_tactical(model, optimizer, dataloader, device, epochs=3, grad_clip=1.0):
    """Train on tactical puzzles with category-based weighting.
    
    Tactical puzzles are crucial for chess strength - they teach the model
    to recognize patterns like forks, pins, and checkmates.
    """
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    model.train()
    
    # Category weights - HEAVILY prioritize checkmate patterns
    category_weights = {
        # Mate puzzles (highest priority - model needs to learn checkmates!)
        'mate_in_one': 10.0,
        'mate_in_two': 8.0,
        'mate_in_three': 6.0,
        'mate_longer': 5.0,
        'backrank_mate': 8.0,
        'smothered_mate': 8.0,
        # Endgame (important for finishing games)
        'endgame': 5.0,
        'promotion': 4.0,
        # Tactics
        'fork': 3.0,
        'double_attack': 3.0,
        'pin': 2.5,
        'skewer': 2.5,
        'discovered': 2.5,
        'sacrifice': 2.0,
        # Default
        'other': 1.0,
        'default': 1.0
    }
    
    for epoch in range(epochs):
        batch_count = 0
        total_loss = 0
        
        for batch in dataloader:
            # Handle variable batch formats
            if len(batch) == 3:
                inputs, policy_targets, value_targets = batch
                categories = ['default'] * inputs.size(0)
            else:
                inputs, policy_targets, value_targets, categories = batch
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (batch)
            policy_logits, value_pred = model(inputs)
            
            # Calculate weighted loss
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred, value_targets)
            
            # Apply category weighting (average weight for batch)
            batch_weight = sum(category_weights.get(c, 1.0) for c in categories) / len(categories)
            
            loss = batch_weight * (2.0 * policy_loss + value_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Tactical epoch {epoch+1}/{epochs}, loss: {avg_loss:.4f}")
    
    return total_loss / batch_count if batch_count > 0 else 0


# ============================================================================
# SELF-PLAY TRAINING UTILITIES
# ============================================================================
def train_on_self_play(model, samples, device, optimizer=None, epochs=3, 
                       batch_size=64, grad_clip=1.0, use_soft_targets=True):
    """Train model on self-play generated samples.
    
    Args:
        model: The neural network
        samples: List of (board_tensor, policy_target, value_target) tuples
        device: Computation device
        optimizer: Optional optimizer (creates new one if None)
        epochs: Number of training epochs
        batch_size: Batch size for training
        grad_clip: Gradient clipping threshold
        use_soft_targets: If True, policy_target should be a distribution
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    if not samples:
        return 0.0
    
    # Create optimizer if not provided
    if optimizer is None:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4
        )
    
    # Prepare data
    boards = torch.stack([torch.tensor(s[0], dtype=torch.float32) for s in samples])
    
    if use_soft_targets:
        # Soft targets (probability distributions)
        policies = torch.from_numpy(np.array([s[1] for s in samples], dtype=np.float32))
    else:
        # Hard targets (indices)
        policies = torch.from_numpy(np.array([s[1] for s in samples], dtype=np.int64))
    
    values = torch.from_numpy(np.array([s[2] for s in samples], dtype=np.float32))
    
    dataset = TensorDataset(boards, policies, values)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss functions
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    model.train()
    total_loss = 0
    batch_count = 0
    
    for epoch in range(epochs):
        for batch_boards, batch_policies, batch_values in dataloader:
            batch_boards = batch_boards.to(device)
            batch_policies = batch_policies.to(device)
            batch_values = batch_values.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(batch_boards)
            
            policy_loss = policy_loss_fn(policy_logits, batch_policies)
            value_loss = value_loss_fn(value_pred, batch_values)
            
            # Equal weighting for self-play (policy and value both important)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
    
    return total_loss / batch_count if batch_count > 0 else 0


