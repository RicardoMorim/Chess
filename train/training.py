import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import json
import os
import sys
import time

from utils import clear_memory, test_tactical_recognition
from data import SelfPlayDataset


def train_batch(model, game_dataloader, puzzle_dataloader, save_path, state_file, epochs=5, processed_games=0, device='cuda'):
    """Train the model on a batch of games and puzzles"""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler() if torch.cuda.is_available() else None
    
    # Use puzzle dataloader as an iterator
    import itertools
    puzzle_iter = itertools.cycle(puzzle_dataloader)
    
    # State loading
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            start_epoch = state.get("last_epoch", 0)
            print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        state = {"processed_games": processed_games, "last_epoch": 0}
        start_epoch = 0

    puzzle_frequency = 1
    puzzle_batch_multiplier = 10  
    policy_weight = 1.5
    value_weight = 1.0
    puzzle_policy_weight = 3.0  
    puzzle_value_weight = 2.0

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        game_batch_count = 0
        
        for game_batch in game_dataloader:
            inputs, policy_targets, value_targets = game_batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler:  # Use mixed precision if available
                # Use autocast for mixed precision
                with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                    policy_logits, value_pred = model(inputs)
                    policy_loss = policy_loss_fn(policy_logits, policy_targets)
                    value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                    loss = policy_weight * policy_loss + value_weight * value_loss
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                policy_logits, value_pred = model(inputs)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                loss = policy_weight * policy_loss + value_weight * value_loss
                
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            game_batch_count += 1

            # Train on puzzle batches
            if game_batch_count % puzzle_frequency == 0:
                for _ in range(puzzle_batch_multiplier):
                    puzzle_batch = next(puzzle_iter)
                    inputs, policy_targets, value_targets = puzzle_batch
                    inputs = inputs.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    if scaler:  # Use mixed precision if available
                        # Use autocast for puzzles too
                        with autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
                            policy_logits, value_pred = model(inputs)
                            policy_loss = policy_loss_fn(policy_logits, policy_targets)
                            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                            loss = puzzle_policy_weight * policy_loss + puzzle_value_weight * value_loss
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        policy_logits, value_pred = model(inputs)
                        policy_loss = policy_loss_fn(policy_logits, policy_targets)
                        value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                        loss = puzzle_policy_weight * policy_loss + puzzle_value_weight * value_loss
                        
                        loss.backward()
                        optimizer.step()
                    
                    total_loss += loss.item()

        # Update scheduler, save checkpoint, etc.
        scheduler.step()
        num_puzzle_batches = len(game_dataloader) // puzzle_frequency * puzzle_batch_multiplier
        avg_loss = total_loss / (len(game_dataloader) + num_puzzle_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        state["last_epoch"] = epoch + 1
        state["processed_games"] = processed_games
        
        torch.save(model.state_dict(), save_path)
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"Checkpoint saved at epoch {epoch + 1}")


def train_tactical(model, optimizer, dataloader, device, epochs=3):
    """Train on tactical puzzles for a specific number of epochs"""
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    model.train()
    
    for epoch in range(epochs):
        batch_count = 0
        total_loss = 0
        
        for batch in dataloader:
            inputs, policy_targets, value_targets = batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            policy_logits, value_pred = model(inputs)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            
            # Higher policy weight for tactical training
            loss = 3.0 * policy_loss + value_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            if batch_count >= 10:  # Limit number of batches for speed
                break
        
        if batch_count > 0:
            avg_loss = total_loss / batch_count
            print(f"Tactical training epoch {epoch+1}, avg loss: {avg_loss:.4f}")
    
    return total_loss / batch_count if batch_count > 0 else 0


def get_optimal_batch_size(model, device, starting_size=32, min_size=8):
    """Find the largest batch size that fits in memory"""
    batch_size = starting_size
    
    while batch_size >= min_size:
        try:
            # Try to create a batch of random data
            dummy_input = torch.randn(batch_size, 20, 8, 8, device=device)
            model(dummy_input)
            dummy_input = None
            clear_memory()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                batch_size //= 2
                clear_memory()
            else:
                raise e
    
    return min_size  # Fallback to minimum size
