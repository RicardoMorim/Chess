"""
Checkmate Training Module
=========================

Specialized training functions for teaching the model checkmate patterns.

Includes:
1. Periodic checkmate reinforcement (runs every N iterations)
2. Optional intensive boot camp mode (one-time startup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import time
import gc

# Import from training module
from training import PolicyLoss, ValueLoss


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
        # Get category from the dataset
        if hasattr(puzzle_dataset, 'categories'):
            category = puzzle_dataset.categories[i]
        elif hasattr(puzzle_dataset, 'puzzles'):
            puzzle = puzzle_dataset.puzzles[i]
            category = puzzle[3] if len(puzzle) >= 4 else 'other'
        else:
            # Try to get from item directly
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
    
    This should be called every few iterations to prevent the model
    from "forgetting" checkmate patterns during other training.
    
    Args:
        model: The neural network
        puzzle_dataset: Full puzzle dataset (will be filtered to mates only)
        device: CUDA/CPU device
        epochs: Number of training epochs (default: 5)
        batch_size: Batch size for training
        lr: Learning rate (higher than normal for focused learning)
    """
    print("\n" + "="*50)
    print("CHECKMATE REINFORCEMENT PHASE")
    print("="*50)
    
    # Filter to mate puzzles only
    mate_indices = filter_mate_puzzles(puzzle_dataset)
    
    if len(mate_indices) < 100:
        print(f"Warning: Only {len(mate_indices)} mate puzzles found. Skipping reinforcement.")
        return 0.0
    
    print(f"Training on {len(mate_indices)} mate puzzles for {epochs} epochs")
    
    # Create subset with only mate puzzles
    mate_subset = Subset(puzzle_dataset, mate_indices)
    
    # DataLoader workers - Windows now supported with proper __main__ guard
    import platform
    num_workers = 2  # Works on Windows with mp.freeze_support() in main
    
    mate_loader = DataLoader(
        mate_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Training setup
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    # Category weights for mate puzzles (all high)
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
            # Handle variable batch formats
            if len(batch) == 4:
                inputs, policy_targets, value_targets, categories = batch
            else:
                inputs, policy_targets, value_targets = batch[:3]
                categories = ['mate_in_one'] * inputs.size(0)  # Assume mate-in-1 if no category
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            policy_logits, value_pred = model(inputs)
            
            # Calculate weighted loss
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred, value_targets)
            
            # Apply category weighting
            batch_weight = sum(category_weights.get(c, 5.0) for c in categories) / len(categories)
            
            # Heavy emphasis on policy (finding the checkmate move)
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
    
    # Cleanup
    del mate_loader, mate_subset
    gc.collect()
    
    return avg_loss


# ============================================================================
# CHECKMATE BOOT CAMP (One-time intensive)
# ============================================================================
def run_checkmate_bootcamp(model, puzzle_dataset, device, save_path,
                           epochs=50, target_accuracy=0.85, batch_size=64):
    """Intensive mate-only training until model achieves target accuracy.
    
    This is an optional one-time intensive training phase that should be
    run at the start to establish strong checkmate pattern recognition.
    
    Includes anti-overfitting measures:
    - Lower learning rate for pretrained models
    - Proper shuffled train/validation split  
    - Early stopping with patience
    - Cosine annealing LR schedule
    - Label smoothing
    
    Args:
        model: The neural network
        puzzle_dataset: Full puzzle dataset (will be filtered to mates only)
        device: CUDA/CPU device
        save_path: Path to save model checkpoints
        epochs: Maximum number of training epochs
        target_accuracy: Stop when this accuracy is reached on mate puzzles
        batch_size: Batch size for training
    """
    import random
    
    print("\n" + "="*60)
    print("CHECKMATE BOOT CAMP - INTENSIVE TRAINING")
    print("="*60)
    print(f"Target accuracy: {target_accuracy:.0%}")
    print(f"Maximum epochs: {epochs}")
    
    # Filter to mate puzzles only
    mate_indices = filter_mate_puzzles(puzzle_dataset)
    
    if len(mate_indices) < 100:
        print(f"Error: Only {len(mate_indices)} mate puzzles found. Need at least 100.")
        return
    
    print(f"Found {len(mate_indices)} mate puzzles for boot camp")
    
    # IMPORTANT: Shuffle indices before splitting to ensure diverse train/val sets
    # Without this, train and val may have different distributions (e.g., all mate-in-1 vs mate-in-3)
    shuffled_indices = mate_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Split into train/validation (85/15) - larger validation for better accuracy estimates
    n_val = max(200, len(shuffled_indices) // 7)  # ~15% for validation
    n_train = len(shuffled_indices) - n_val
    
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:]
    
    print(f"  Train: {n_train} puzzles, Validation: {n_val} puzzles")
    
    train_subset = Subset(puzzle_dataset, train_indices)
    val_subset = Subset(puzzle_dataset, val_indices)
    
    import platform
    num_workers = 2  # Works on Windows with mp.freeze_support() in main
    
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True  # Avoid small batches that can destabilize training
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # ANTI-OVERFITTING: Freeze backbone for first phase
    # This prevents the 18M param model from overfitting on 58K puzzles
    freeze_epochs = 5  # Train only heads for first 5 epochs
    if hasattr(model, 'freeze_backbone'):
        model.freeze_backbone()
        params_info = model.get_trainable_params()
        print(f"  Trainable params: {params_info['trainable']:,} / {params_info['total']:,} "
              f"({100*params_info['trainable']/params_info['total']:.1f}%)")
    
    # Training setup - VERY LOW LR for fine-tuning pretrained models
    # Lower is better to prevent destroying learned features
    initial_lr = 0.0005  # Much gentler for pretrained models (was 0.002)
    
    optimizer = torch.optim.AdamW(  # AdamW works better for fine-tuning
        filter(lambda p: p.requires_grad, model.parameters()),  # Only trainable params
        lr=initial_lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Cosine annealing with warm restarts prevents getting stuck in local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=10,  # Restart every 10 epochs
        T_mult=2,  # Double the period after each restart
        eta_min=1e-6
    )
    
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    # Label smoothing to prevent overconfidence
    label_smoothing = 0.1
    
    best_accuracy = 0.0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    patience = 8  # Early stopping patience
    backbone_unfrozen = False  # Track if we've unfrozen the backbone
    
    start_time = time.time()

    
    # Load best model state for potential restoration
    import copy
    best_model_state = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        # Unfreeze backbone after initial frozen epochs
        if epoch >= freeze_epochs and not backbone_unfrozen and hasattr(model, 'unfreeze_backbone'):
            model.unfreeze_backbone()
            backbone_unfrozen = True
            # Recreate optimizer to include all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=initial_lr * 0.5,  # Use lower LR for unfrozen backbone
                weight_decay=1e-4,
                betas=(0.9, 0.999)
            )
            print(f"  Recreated optimizer with all parameters, LR={initial_lr * 0.5:.6f}")
        
        # Training phase with dropout enabled
        model.train()
        epoch_loss = 0
        epoch_policy_loss = 0
        epoch_batches = 0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            if len(batch) == 4:
                inputs, policy_targets, value_targets, _ = batch
            else:
                inputs, policy_targets, value_targets = batch[:3]
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_pred = model(inputs)
            
            # Apply label smoothing to policy loss
            # This prevents the model from being overconfident
            n_classes = policy_logits.size(1)
            smooth_targets = torch.zeros_like(policy_logits)
            smooth_targets.fill_(label_smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, policy_targets.unsqueeze(1), 1.0 - label_smoothing)
            
            policy_loss = -(smooth_targets * F.log_softmax(policy_logits, dim=1)).sum(dim=1).mean()
            value_loss = value_loss_fn(value_pred, value_targets)
            
            # Heavy policy emphasis for checkmate moves
            loss = 3.0 * policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_batches += 1
            
            # Track training accuracy
            predictions = policy_logits.argmax(dim=1)
            train_correct += (predictions == policy_targets).sum().item()
            train_total += inputs.size(0)
        
        scheduler.step()
        
        train_accuracy = train_correct / max(1, train_total)
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    inputs, policy_targets, value_targets, _ = batch
                else:
                    inputs, policy_targets, value_targets = batch[:3]
                
                inputs = inputs.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                
                policy_logits, value_pred = model(inputs)
                predictions = policy_logits.argmax(dim=1)
                
                # Standard loss for validation (no label smoothing)
                p_loss = policy_loss_fn(policy_logits, policy_targets)
                v_loss = value_loss_fn(value_pred, value_targets)
                batch_loss = 3.0 * p_loss + v_loss
                
                val_loss += batch_loss.item()
                val_batches += 1
                val_correct += (predictions == policy_targets).sum().item()
                val_total += inputs.size(0)
        
        val_accuracy = val_correct / max(1, val_total)
        avg_train_loss = epoch_loss / max(1, epoch_batches)
        avg_val_loss = val_loss / max(1, val_batches)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Calculate generalization gap (train_acc - val_acc indicates overfitting)
        gen_gap = train_accuracy - val_accuracy
        overfit_warning = " ⚠️ OVERFITTING" if gen_gap > 0.15 else ""
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
              f"train_acc={train_accuracy:.2%}, val_acc={val_accuracy:.2%}, lr={current_lr:.6f}{overfit_warning}")
        
        # Save if best validation accuracy (not training accuracy!)
        improved = False
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best! Saved to {save_path}")
            improved = True
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Check if target reached
        if val_accuracy >= target_accuracy:
            print(f"\n🎉 Target accuracy {target_accuracy:.0%} reached!")
            break
        
        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"\n⏹️ Early stopping: No improvement for {patience} epochs")
            print(f"  Restoring best model (val_acc={best_accuracy:.2%})")
            model.load_state_dict(best_model_state)
            break
        
        # If severe overfitting detected, reduce LR more aggressively
        if gen_gap > 0.20 and epoch > 5:
            print(f"  📉 Severe overfitting detected, reducing LR")
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
    
    elapsed = time.time() - start_time
    print(f"\nBoot camp complete in {elapsed/60:.1f} minutes")
    print(f"Best validation accuracy: {best_accuracy:.2%}")
    print("="*60 + "\n")
    
    # Ensure best model is loaded
    model.load_state_dict(best_model_state)
    
    # Ensure backbone is unfrozen for subsequent training
    if hasattr(model, 'unfreeze_backbone'):
        model.unfreeze_backbone()
    
    # Cleanup
    del train_loader, val_loader, train_subset, val_subset
    gc.collect()
    
    return best_accuracy


# ============================================================================
# TESTING
# ============================================================================
def test_checkmate_accuracy(model, puzzle_dataset, device, num_samples=500):
    """Test model accuracy on checkmate puzzles.
    
    Returns accuracy (0.0-1.0) on mate puzzles.
    """
    mate_indices = filter_mate_puzzles(puzzle_dataset)
    
    if len(mate_indices) == 0:
        return 0.0
    
    # Sample if too many
    if len(mate_indices) > num_samples:
        import random
        mate_indices = random.sample(mate_indices, num_samples)
    
    mate_subset = Subset(puzzle_dataset, mate_indices)
    
    import platform
    num_workers = 2  # Works on Windows with mp.freeze_support() in main
    
    loader = DataLoader(
        mate_subset,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers
    )
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                inputs, policy_targets, _, _ = batch
            else:
                inputs, policy_targets, _ = batch[:3]
            
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            
            policy_logits, _ = model(inputs)
            predictions = policy_logits.argmax(dim=1)
            
            correct += (predictions == policy_targets).sum().item()
            total += inputs.size(0)
    
    return correct / max(1, total)
