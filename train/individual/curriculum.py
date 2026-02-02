"""
3-Phase Curriculum Training
===========================

Phase 1: Puzzle Bootcamp - Bootstrap tactical knowledge from puzzles
Phase 2: Transition - Blend puzzle knowledge with initial self-play
Phase 3: Pure Self-Play - Convergence loop (runs indefinitely)
"""

import os
import time
import json

import torch
from torch.utils.data import DataLoader, ConcatDataset

# Import from core
from core.models import create_model
from core.data import PuzzleDataset, SelfPlayDataset, load_lichess_puzzles
from core.constants import (
    MODEL_CONFIG, CURRICULUM_CONFIG, TRAINING_CONFIG, HARDWARE_CONFIG
)
from core.utils import clear_memory, test_tactical_recognition

# Import checkmate training from this module
from .checkmate import run_checkmate_bootcamp, run_checkmate_reinforcement


def _get_device():
    """Get the computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _create_dataloader(dataset, batch_size, shuffle=True, num_workers=4):
    """Create a DataLoader with sensible defaults."""
    device = _get_device()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda'
    )


# ============================================================================
# PHASE 1: PUZZLE BOOTCAMP
# ============================================================================
def phase1_puzzle_bootcamp(model, variant, checkpoint_dir, skip_bootcamp=False):
    """
    Phase 1: Isolated puzzle training to bootstrap tactical knowledge.
    
    Args:
        model: The neural network
        variant: Model variant (baseline, attack, est)
        checkpoint_dir: Where to save checkpoints
        skip_bootcamp: Skip intensive checkmate bootcamp
    
    Returns:
        Trained model
    """
    device = _get_device()
    
    print("\n" + "="*80)
    print("PHASE 1: PUZZLE BOOTCAMP (ISOLATED)")
    print("="*80)
    print("Goal: Bootstrap tactical priors and checkmate patterns")
    print(f"Epochs: {CURRICULUM_CONFIG['phase1_epochs']}")
    print(f"Batch size: {CURRICULUM_CONFIG['phase1_batch_size']}")
    print(f"Target accuracy: {CURRICULUM_CONFIG['phase1_target_accuracy']:.1%}")
    print("="*80 + "\n")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load puzzles
    print("Loading Lichess puzzle database...")
    puzzles = load_lichess_puzzles()
    print(f"✓ Loaded {len(puzzles)} puzzles\n")
    
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles,
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    # Optional checkmate bootcamp
    if CURRICULUM_CONFIG.get('phase1_checkmate_bootcamp', True) and not skip_bootcamp:
        print("Running Checkmate Bootcamp (intensive)...\n")
        bootcamp_path = os.path.join(checkpoint_dir, "phase1_bootcamp.pt")
        run_checkmate_bootcamp(
            model=model,
            puzzle_dataset=puzzle_dataset,
            device=device,
            save_path=bootcamp_path,
            epochs=10,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size']
        )
        print(f"✓ Checkmate bootcamp complete, saved to {bootcamp_path}\n")
    
    # Main puzzle training
    puzzle_loader = _create_dataloader(
        puzzle_dataset,
        batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
        shuffle=True,
        num_workers=HARDWARE_CONFIG.get('dataloader_workers', 4)
    )
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CURRICULUM_CONFIG['phase1_epochs']
    )
    
    best_accuracy = 0.0
    
    for epoch in range(CURRICULUM_CONFIG['phase1_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(puzzle_loader):
            states, policy_targets, value_targets = batch[:3]
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_preds = model(states)
            
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
            value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            
            predictions = torch.argmax(policy_logits, dim=1)
            targets = policy_targets if policy_targets.dim() == 1 else torch.argmax(policy_targets, dim=1)
            correct += (predictions == targets).sum().item()
            total += states.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(puzzle_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{CURRICULUM_CONFIG['phase1_epochs']}: "
              f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(checkpoint_dir, "phase1_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best accuracy: {accuracy:.2%}")
        
        if accuracy >= CURRICULUM_CONFIG['phase1_target_accuracy']:
            print(f"\n✓ Target accuracy {CURRICULUM_CONFIG['phase1_target_accuracy']:.1%} reached!")
            break
    
    final_path = os.path.join(checkpoint_dir, "phase1_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nPhase 1 complete! Final accuracy: {best_accuracy:.2%}")
    print(f"Model saved to {final_path}\n")
    
    return model


# ============================================================================
# PHASE 2: TRANSITION
# ============================================================================
def phase2_transition(model, variant, checkpoint_dir, generate_games_fn=None):
    """
    Phase 2: Brief transition to self-play with initial game generation.
    
    Args:
        model: The neural network
        variant: Model variant
        checkpoint_dir: Where to save checkpoints
        generate_games_fn: Function to generate self-play games (optional)
    
    Returns:
        Trained model
    """
    device = _get_device()
    
    print("\n" + "="*80)
    print("PHASE 2: TRANSITION (BRIEF HANDOFF)")
    print("="*80)
    print("Goal: Generate initial self-play games and blend knowledge")
    print(f"Epochs: {CURRICULUM_CONFIG['phase2_epochs']}")
    print(f"Games to generate: {CURRICULUM_CONFIG['phase2_games']}")
    print(f"MCTS simulations: {CURRICULUM_CONFIG['phase2_mcts_sims']}")
    print("="*80 + "\n")
    
    replay_dir = os.path.join(checkpoint_dir, "replay_buffer")
    os.makedirs(replay_dir, exist_ok=True)
    
    # Generate self-play games
    if generate_games_fn is not None:
        print(f"Generating {CURRICULUM_CONFIG['phase2_games']} self-play games...")
        games = generate_games_fn(
            model=model,
            device=device,
            num_games=CURRICULUM_CONFIG['phase2_games'],
            num_simulations=CURRICULUM_CONFIG['phase2_mcts_sims']
        )
        print(f"✓ Generated {len(games)} games\n")
    else:
        print("⚠ No game generation function provided, using empty game list")
        games = []
    
    # Save games
    replay_path = os.path.join(replay_dir, "phase2_games.json")
    with open(replay_path, 'w') as f:
        json.dump(games, f)
    
    if not games:
        print("Skipping training (no games generated)")
        return model
    
    # Create datasets
    selfplay_dataset = SelfPlayDataset(
        games=games,
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    puzzles = load_lichess_puzzles()
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles[:10000],
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    combined_dataset = ConcatDataset([selfplay_dataset, puzzle_dataset])
    
    dataloader = _create_dataloader(
        combined_dataset,
        batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
        shuffle=True
    )
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    for epoch in range(CURRICULUM_CONFIG['phase2_epochs']):
        model.train()
        total_loss = 0
        
        for states, policy_targets, value_targets in dataloader:
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_preds = model(states)
            
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
            value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{CURRICULUM_CONFIG['phase2_epochs']}: Loss={avg_loss:.4f}")
    
    transition_path = os.path.join(checkpoint_dir, "phase2_final.pt")
    torch.save(model.state_dict(), transition_path)
    print(f"\nPhase 2 complete! Model saved to {transition_path}\n")
    
    return model


# ============================================================================
# PHASE 3: PURE SELF-PLAY
# ============================================================================
def phase3_pure_selfplay(model, variant, checkpoint_dir, 
                         generate_games_fn=None, puzzle_dataset=None):
    """
    Phase 3: Pure self-play convergence loop (runs indefinitely).
    
    Args:
        model: The neural network
        variant: Model variant
        checkpoint_dir: Where to save checkpoints
        generate_games_fn: Function to generate self-play games
        puzzle_dataset: For checkmate reinforcement (optional)
    
    Returns:
        Trained model (when interrupted)
    """
    device = _get_device()
    
    print("\n" + "="*80)
    print("PHASE 3: PURE SELF-PLAY (CONVERGENCE LOOP)")
    print("="*80)
    print("Goal: Converge to optimal play through self-improvement")
    print(f"Games per iteration: {CURRICULUM_CONFIG['phase3_games_per_iteration']}")
    print(f"Training epochs: {CURRICULUM_CONFIG['phase3_training_epochs']}")
    print(f"Batch size: {CURRICULUM_CONFIG['phase3_batch_size']}")
    print(f"MCTS simulations: {CURRICULUM_CONFIG['phase3_mcts_sims']}")
    print("Note: This phase runs indefinitely (Ctrl+C to stop)")
    print("="*80 + "\n")
    
    replay_dir = os.path.join(checkpoint_dir, "replay_buffer")
    os.makedirs(replay_dir, exist_ok=True)
    
    iteration = 0
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    try:
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"SELF-PLAY ITERATION {iteration}")
            print(f"{'='*80}\n")
            
            # Generate games
            if generate_games_fn is not None:
                print(f"Generating {CURRICULUM_CONFIG['phase3_games_per_iteration']} games...")
                start_time = time.time()
                
                games = generate_games_fn(
                    model=model,
                    device=device,
                    num_games=CURRICULUM_CONFIG['phase3_games_per_iteration'],
                    num_simulations=CURRICULUM_CONFIG['phase3_mcts_sims']
                )
                
                gen_time = time.time() - start_time
                print(f"✓ Generated {len(games)} games in {gen_time:.1f}s\n")
            else:
                print("⚠ No game generation function, skipping")
                continue
            
            # Save games
            replay_path = os.path.join(replay_dir, f"iteration_{iteration:04d}.json")
            with open(replay_path, 'w') as f:
                json.dump(games, f)
            
            # Train
            selfplay_dataset = SelfPlayDataset(
                games=games,
                input_channels=MODEL_CONFIG[variant]['input_channels']
            )
            
            dataloader = _create_dataloader(
                selfplay_dataset,
                batch_size=CURRICULUM_CONFIG['phase3_batch_size'],
                shuffle=True
            )
            
            print(f"Training for {CURRICULUM_CONFIG['phase3_training_epochs']} epochs...")
            for epoch in range(CURRICULUM_CONFIG['phase3_training_epochs']):
                model.train()
                total_loss = 0
                
                for states, policy_targets, value_targets in dataloader:
                    states = states.to(device)
                    policy_targets = policy_targets.to(device)
                    value_targets = value_targets.to(device)
                    
                    optimizer.zero_grad()
                    
                    policy_logits, value_preds = model(states)
                    
                    policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
                    value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
                    loss = policy_loss + value_loss
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
            
            # Save checkpoint
            iter_path = os.path.join(checkpoint_dir, f"phase3_iter_{iteration:04d}.pt")
            torch.save(model.state_dict(), iter_path)
            print(f"\n✓ Iteration {iteration} complete, saved to {iter_path}")
            
            # Periodic checkmate reinforcement
            checkmate_interval = CURRICULUM_CONFIG.get('phase3_checkmate_interval', 5)
            if checkmate_interval > 0 and iteration % checkmate_interval == 0 and puzzle_dataset:
                print("\nRunning checkmate reinforcement...")
                run_checkmate_reinforcement(
                    model=model,
                    puzzle_dataset=puzzle_dataset,
                    device=device,
                    epochs=5,
                    batch_size=CURRICULUM_CONFIG['phase3_batch_size']
                )
            
            # Periodic evaluation
            eval_interval = CURRICULUM_CONFIG.get('phase3_evaluation_interval', 10)
            if eval_interval > 0 and iteration % eval_interval == 0:
                print("\nRunning tactical evaluation...")
                test_tactical_recognition(model, device)
            
            # Memory cleanup
            clear_memory()
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        final_path = os.path.join(checkpoint_dir, "phase3_interrupted.pt")
        torch.save(model.state_dict(), final_path)
        print(f"Model saved to {final_path}")
    
    return model
