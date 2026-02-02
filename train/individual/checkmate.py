"""
Checkmate Training
==================

Specialized training for checkmate pattern recognition.
- Reinforcement: Periodic refresher (every N iterations)
- Bootcamp: One-time intensive training at startup
"""

import time
import gc
import copy
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# Import from core
from core.training import PolicyLoss, ValueLoss


# ============================================================================
# MATE PUZZLE FILTERING
# ============================================================================
MATE_CATEGORIES = {
    'mate_in_one', 'mate_in_two', 'mate_in_three', 
    'mate_longer', 'backrank_mate', 'smothered_mate'
}


def filter_mate_puzzles(puzzle_dataset):
    """Filter puzzle dataset to only include checkmate puzzles.
    
    Returns indices of mate puzzles for use with Subset.
    """
    mate_indices = []
    
    for i in range(len(puzzle_dataset)):
        if hasattr(puzzle_dataset, 'categories'):
            category = puzzle_dataset.categories[i]
        elif hasattr(puzzle_dataset, 'puzzles'):
            puzzle = puzzle_dataset.puzzles[i]
            category = puzzle[3] if len(puzzle) >= 4 else 'other'
        else:
            try:
                item = puzzle_dataset[i]
                category = item[3] if len(item) >= 4 else 'other'
            except:
                continue
        
        if category in MATE_CATEGORIES:
            mate_indices.append(i)
    
    return mate_indices


# ============================================================================
# CHECKMATE REINFORCEMENT (Periodic)
# ============================================================================
def run_checkmate_reinforcement(model, puzzle_dataset, device, epochs=5, 
                                 batch_size=64, lr=0.002):
    """Periodic mate-only training to reinforce checkmate patterns.
    
    Should be called every few iterations to prevent the model
    from "forgetting" checkmate patterns during other training.
    """
    print("\n" + "="*50)
    print("CHECKMATE REINFORCEMENT PHASE")
    print("="*50)
    
    mate_indices = filter_mate_puzzles(puzzle_dataset)
    
    if len(mate_indices) < 100:
        print(f"Warning: Only {len(mate_indices)} mate puzzles found. Skipping.")
        return 0.0
    
    print(f"Training on {len(mate_indices)} mate puzzles for {epochs} epochs")
    
    mate_subset = Subset(puzzle_dataset, mate_indices)
    
    mate_loader = DataLoader(
        mate_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == 'cuda'
    )
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    category_weights = {
        'mate_in_one': 10.0,
        'mate_in_two': 8.0,
        'mate_in_three': 6.0,
        'mate_longer': 5.0,
        'backrank_mate': 8.0,
        'smothered_mate': 8.0,
    }
    
    model.train()
    total_loss = 0
    batch_count = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_batches = 0
        
        for batch in mate_loader:
            if len(batch) == 4:
                inputs, policy_targets, value_targets, categories = batch
            else:
                inputs, policy_targets, value_targets = batch[:3]
                categories = ['mate_in_one'] * inputs.size(0)
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(inputs)
            
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred, value_targets)
            
            batch_weight = sum(category_weights.get(c, 5.0) for c in categories) / len(categories)
            loss = batch_weight * (3.0 * policy_loss + value_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            total_loss += loss.item()
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / max(1, epoch_batches)
        print(f"  Epoch {epoch+1}/{epochs}: loss = {avg_epoch_loss:.4f}")
    
    elapsed = time.time() - start_time
    avg_loss = total_loss / max(1, batch_count)
    
    print(f"Checkmate reinforcement complete: {elapsed:.1f}s, avg loss: {avg_loss:.4f}")
    print("="*50 + "\n")
    
    del mate_loader, mate_subset
    gc.collect()
    
    return avg_loss


# ============================================================================
# CHECKMATE BOOT CAMP (One-time intensive)
# ============================================================================
def run_checkmate_bootcamp(model, puzzle_dataset, device, save_path=None,
                           epochs=50, target_accuracy=0.85, batch_size=64):
    """Intensive mate-only training until model achieves target accuracy.
    
    Includes anti-overfitting measures:
    - Lower learning rate for pretrained models
    - Proper shuffled train/validation split  
    - Early stopping with patience
    - Cosine annealing LR schedule
    - Label smoothing
    """
    print("\n" + "="*60)
    print("CHECKMATE BOOT CAMP - INTENSIVE TRAINING")
    print("="*60)
    print(f"Target accuracy: {target_accuracy:.0%}")
    print(f"Maximum epochs: {epochs}")
    
    mate_indices = filter_mate_puzzles(puzzle_dataset)
    
    if len(mate_indices) < 100:
        print(f"Error: Only {len(mate_indices)} mate puzzles found. Need at least 100.")
        return 0.0
    
    print(f"Found {len(mate_indices)} mate puzzles for boot camp")
    
    # Shuffle and split
    shuffled_indices = mate_indices.copy()
    random.shuffle(shuffled_indices)
    
    n_val = max(200, len(shuffled_indices) // 7)
    n_train = len(shuffled_indices) - n_val
    
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    
    print(f"  Train: {n_train} puzzles, Validation: {n_val} puzzles")
    
    train_subset = Subset(puzzle_dataset, train_indices)
    val_subset = Subset(puzzle_dataset, val_indices)
    
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=device.type == 'cuda', drop_last=True
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=device.type == 'cuda'
    )
    
    # Low LR for fine-tuning
    initial_lr = 0.0005
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    label_smoothing = 0.1
    best_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 8
    
    start_time = time.time()
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_batches = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if len(batch) >= 3:
                inputs, policy_targets, value_targets = batch[:3]
            else:
                continue
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(inputs)
            
            # Label smoothing
            n_classes = policy_logits.size(1)
            smooth_targets = torch.zeros_like(policy_logits)
            smooth_targets.fill_(label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, policy_targets.unsqueeze(1), 1.0 - label_smoothing)
            
            policy_loss = -(smooth_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = value_loss_fn(value_pred, value_targets)
            
            loss = 3.0 * policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_batches += 1
            
            predictions = policy_logits.argmax(dim=1)
            train_correct += (predictions == policy_targets).sum().item()
            train_total += inputs.size(0)
        
        scheduler.step()
        train_accuracy = train_correct / max(1, train_total)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss_sum = 0
        
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
                
                policy_loss = F.cross_entropy(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred, value_targets)
                val_loss_sum += (policy_loss + value_loss).item()
                
                predictions = policy_logits.argmax(dim=1)
                val_correct += (predictions == policy_targets).sum().item()
                val_total += inputs.size(0)
        
        val_accuracy = val_correct / max(1, val_total)
        avg_val_loss = val_loss_sum / max(1, len(val_loader))
        avg_train_loss = epoch_loss / max(1, epoch_batches)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={avg_train_loss:.4f}, Train Acc={train_accuracy:.2%}, "
              f"Val Acc={val_accuracy:.2%}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Check for improvement
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ New best accuracy: {val_accuracy:.2%}")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: No improvement for {patience} epochs")
            break
        
        # Target reached
        if val_accuracy >= target_accuracy:
            print(f"\n✓ Target accuracy {target_accuracy:.1%} reached!")
            break
    
    # Restore best model
    model.load_state_dict(best_model_state)
    
    elapsed = time.time() - start_time
    print(f"\nCheckmate bootcamp complete: {elapsed:.1f}s, best accuracy: {best_accuracy:.2%}")
    print("="*60 + "\n")
    
    del train_loader, val_loader
    gc.collect()
    
    return best_accuracy
