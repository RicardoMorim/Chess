"""
3-Phase Curriculum Training for Chess AI
=========================================

Phase 1: Puzzle Bootcamp (Isolated)
    - Train exclusively on tactical puzzles
    - Focus on checkmate patterns and tactics
    - Target: 75% tactical accuracy
    
Phase 2: Transition (Brief)
    - Generate initial self-play games
    - Mix puzzle knowledge with self-play data
    - Target: Smooth handoff to pure self-play

Phase 3: Pure Self-Play (Convergence Loop)
    - Generate high-quality games via MCTS
    - Train on self-play only
    - Optional: Periodic checkmate reinforcement
    - Loop forever until convergence

Hardware: RTX 5080 16GB + Intel Ultra 9 24-core
"""

import os
import sys
import time
import json
import torch
import argparse
import multiprocessing as mp
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset

# Model and data
from models import create_model, load_model_with_compatibility
from data import PuzzleDataset, SelfPlayDataset, load_lichess_puzzles
from constants import (
    MODEL_CONFIG, VALID_VARIANTS,
    CURRICULUM_CONFIG, TRAINING_CONFIG,
    MCTS_CONFIG, SELF_PLAY_CONFIG,
    HARDWARE_CONFIG
)

# Training utilities
from utils import clear_memory, test_tactical_recognition, model_summary
from self_play import generate_self_play_games
from training import train_on_self_play
from checkmate_training import run_checkmate_bootcamp, run_checkmate_reinforcement

# Optional advanced features
try:
    from training_utils import CheckpointManager, TensorBoardLogger, StockfishEvaluator
    HAS_TRAINING_UTILS = True
except ImportError:
    HAS_TRAINING_UTILS = False
    print("⚠ Advanced training utilities not available")

try:
    from optimizations import create_optimized_dataloader, aggressive_memory_cleanup
    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False
    print("⚠ Optimizations not available")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Enable TF32 for Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="3-Phase Curriculum Chess AI Training")
    
    # Model selection
    parser.add_argument("--variant", default="baseline", choices=VALID_VARIANTS,
                        help="Model variant: baseline, attack, or est")
    
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints (default: ./checkpoints_{variant})")
    
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")
    
    # Phase control
    parser.add_argument("--start-phase", type=int, default=1, choices=[1, 2, 3],
                        help="Start from phase (1=Puzzle Bootcamp, 2=Transition, 3=Self-Play)")
    
    parser.add_argument("--skip-bootcamp", action="store_true",
                        help="Skip checkmate bootcamp in Phase 1")
    
    # Optional features
    parser.add_argument("--tensorboard", action="store_true",
                        help="Enable TensorBoard logging")
    
    parser.add_argument("--stockfish-eval", action="store_true",
                        help="Evaluate with Stockfish (optional)")
    
    parser.add_argument("--stockfish-path", type=str, default=None,
                        help="Path to Stockfish executable")
    
    args = parser.parse_args()
    
    # Set default checkpoint directory
    if args.checkpoint_dir is None:
        args.checkpoint_dir = f"./checkpoints_{args.variant}"
    
    return args


def phase1_puzzle_bootcamp(model, variant, checkpoint_dir, args):
    """
    Phase 1: Isolated puzzle training to bootstrap tactical knowledge
    """
    print("\n" + "="*80)
    print("PHASE 1: PUZZLE BOOTCAMP (ISOLATED)")
    print("="*80)
    print("Goal: Bootstrap tactical priors and checkmate patterns")
    print(f"Epochs: {CURRICULUM_CONFIG['phase1_epochs']}")
    print(f"Batch size: {CURRICULUM_CONFIG['phase1_batch_size']}")
    print(f"Target accuracy: {CURRICULUM_CONFIG['phase1_target_accuracy']:.1%}")
    print("="*80 + "\n")
    
    # Optional: Intensive checkmate bootcamp first
    if CURRICULUM_CONFIG['phase1_checkmate_bootcamp'] and not args.skip_bootcamp:
        print("Running Checkmate Bootcamp (intensive)...\n")
        run_checkmate_bootcamp(
            model=model,
            device=device,
            input_channels=MODEL_CONFIG[variant]['input_channels'],
            epochs=10,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size']
        )
        
        # Save bootcamp checkpoint
        bootcamp_path = os.path.join(checkpoint_dir, "phase1_bootcamp.pt")
        torch.save(model.state_dict(), bootcamp_path)
        print(f"✓ Checkmate bootcamp complete, saved to {bootcamp_path}\n")
    
    # Load puzzle datasets
    print("Loading Lichess puzzle database...")
    puzzles = load_lichess_puzzles()
    print(f"✓ Loaded {len(puzzles)} puzzles\n")
    
    # Create puzzle dataset
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles,
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    # Create dataloader
    if HAS_OPTIMIZATIONS:
        puzzle_loader = create_optimized_dataloader(
            puzzle_dataset,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
            shuffle=True,
            num_workers=HARDWARE_CONFIG['dataloader_workers'],
            pin_memory=HARDWARE_CONFIG['pin_memory']
        )
    else:
        puzzle_loader = DataLoader(
            puzzle_dataset,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Setup scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=CURRICULUM_CONFIG['phase1_epochs']
    )
    
    # Training loop
    best_accuracy = 0.0
    for epoch in range(CURRICULUM_CONFIG['phase1_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (states, policy_targets, value_targets) in enumerate(puzzle_loader):
            states = states.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            policy_logits, value_preds = model(states)
            
            # Loss calculation
            policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
            value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
            loss = policy_loss + value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            predictions = torch.argmax(policy_logits, dim=1)
            targets = torch.argmax(policy_targets, dim=1)
            correct += (predictions == targets).sum().item()
            total += states.size(0)
        
        scheduler.step()
        
        avg_loss = total_loss / len(puzzle_loader)
        accuracy = correct / total
        
        print(f"Epoch {epoch+1}/{CURRICULUM_CONFIG['phase1_epochs']}: "
              f"Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}, LR={scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = os.path.join(checkpoint_dir, "phase1_best.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ New best accuracy: {accuracy:.2%}, saved to {best_path}")
        
        # Check if target reached
        if accuracy >= CURRICULUM_CONFIG['phase1_target_accuracy']:
            print(f"\n✓ Target accuracy {CURRICULUM_CONFIG['phase1_target_accuracy']:.1%} reached!")
            break
    
    # Final save
    final_path = os.path.join(checkpoint_dir, "phase1_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nPhase 1 complete! Final accuracy: {best_accuracy:.2%}")
    print(f"Model saved to {final_path}\n")
    
    return model


def phase2_transition(model, variant, checkpoint_dir, args):
    """
    Phase 2: Brief transition to self-play with initial game generation
    """
    print("\n" + "="*80)
    print("PHASE 2: TRANSITION (BRIEF HANDOFF)")
    print("="*80)
    print("Goal: Generate initial self-play games and blend knowledge")
    print(f"Epochs: {CURRICULUM_CONFIG['phase2_epochs']}")
    print(f"Games to generate: {CURRICULUM_CONFIG['phase2_games']}")
    print(f"MCTS simulations: {CURRICULUM_CONFIG['phase2_mcts_sims']}")
    print("="*80 + "\n")
    
    # Generate initial self-play games
    print(f"Generating {CURRICULUM_CONFIG['phase2_games']} self-play games...")
    
    replay_dir = os.path.join(checkpoint_dir, "replay_buffer")
    os.makedirs(replay_dir, exist_ok=True)
    
    games = generate_self_play_games(
        model=model,
        device=device,
        input_channels=MODEL_CONFIG[variant]['input_channels'],
        num_games=CURRICULUM_CONFIG['phase2_games'],
        num_simulations=CURRICULUM_CONFIG['phase2_mcts_sims'],
        num_workers=HARDWARE_CONFIG['selfplay_workers']
    )
    
    print(f"✓ Generated {len(games)} games\n")
    
    # Save games to replay buffer
    replay_path = os.path.join(replay_dir, "phase2_games.json")
    with open(replay_path, 'w') as f:
        json.dump(games, f)
    print(f"✓ Saved games to {replay_path}\n")
    
    # Create self-play dataset
    selfplay_dataset = SelfPlayDataset(
        games=games,
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    # Load puzzles for blending
    puzzles = load_lichess_puzzles()
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles[:10000],  # Subset for transition
        input_channels=MODEL_CONFIG[variant]['input_channels']
    )
    
    # Mix datasets (50/50)
    combined_dataset = ConcatDataset([selfplay_dataset, puzzle_dataset])
    
    # Create dataloader
    if HAS_OPTIMIZATIONS:
        dataloader = create_optimized_dataloader(
            combined_dataset,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
            shuffle=True,
            num_workers=HARDWARE_CONFIG['dataloader_workers']
        )
    else:
        dataloader = DataLoader(
            combined_dataset,
            batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
            shuffle=True,
            num_workers=4
        )
    
    # Setup optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    # Training loop
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
    
    # Save transition checkpoint
    transition_path = os.path.join(checkpoint_dir, "phase2_final.pt")
    torch.save(model.state_dict(), transition_path)
    print(f"\nPhase 2 complete! Model saved to {transition_path}\n")
    
    return model


def phase3_pure_selfplay(model, variant, checkpoint_dir, args):
    """
    Phase 3: Pure self-play convergence loop (runs forever)
    """
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
    
    # Setup optimizer
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
            
            # Generate self-play games
            print(f"Generating {CURRICULUM_CONFIG['phase3_games_per_iteration']} games...")
            start_time = time.time()
            
            games = generate_self_play_games(
                model=model,
                device=device,
                input_channels=MODEL_CONFIG[variant]['input_channels'],
                num_games=CURRICULUM_CONFIG['phase3_games_per_iteration'],
                num_simulations=CURRICULUM_CONFIG['phase3_mcts_sims'],
                num_workers=HARDWARE_CONFIG['selfplay_workers']
            )
            
            gen_time = time.time() - start_time
            print(f"✓ Generated {len(games)} games in {gen_time:.1f}s\n")
            
            # Save games
            replay_path = os.path.join(replay_dir, f"iteration_{iteration:04d}.json")
            with open(replay_path, 'w') as f:
                json.dump(games, f)
            
            # Create dataset
            selfplay_dataset = SelfPlayDataset(
                games=games,
                input_channels=MODEL_CONFIG[variant]['input_channels']
            )
            
            # Create dataloader
            if HAS_OPTIMIZATIONS:
                dataloader = create_optimized_dataloader(
                    selfplay_dataset,
                    batch_size=CURRICULUM_CONFIG['phase3_batch_size'],
                    shuffle=True,
                    num_workers=HARDWARE_CONFIG['dataloader_workers']
                )
            else:
                dataloader = DataLoader(
                    selfplay_dataset,
                    batch_size=CURRICULUM_CONFIG['phase3_batch_size'],
                    shuffle=True,
                    num_workers=4
                )
            
            # Train on self-play data
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
            
            # Save iteration checkpoint
            iter_path = os.path.join(checkpoint_dir, f"phase3_iter_{iteration:04d}.pt")
            torch.save(model.state_dict(), iter_path)
            print(f"\n✓ Iteration {iteration} complete, saved to {iter_path}")
            
            # Periodic checkmate reinforcement
            if (CURRICULUM_CONFIG['phase3_checkmate_interval'] > 0 and 
                iteration % CURRICULUM_CONFIG['phase3_checkmate_interval'] == 0):
                print("\nRunning checkmate reinforcement...")
                run_checkmate_reinforcement(
                    model=model,
                    device=device,
                    input_channels=MODEL_CONFIG[variant]['input_channels'],
                    epochs=5,
                    batch_size=CURRICULUM_CONFIG['phase3_batch_size']
                )
            
            # Optional: Evaluation
            if (CURRICULUM_CONFIG['phase3_evaluation_interval'] > 0 and
                iteration % CURRICULUM_CONFIG['phase3_evaluation_interval'] == 0):
                print("\nRunning tactical evaluation...")
                test_tactical_recognition(
                    model=model,
                    device=device,
                    input_channels=MODEL_CONFIG[variant]['input_channels']
                )
            
            # Memory cleanup
            if HAS_OPTIMIZATIONS:
                aggressive_memory_cleanup()
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        final_path = os.path.join(checkpoint_dir, "phase3_interrupted.pt")
        torch.save(model.state_dict(), final_path)
        print(f"Model saved to {final_path}")


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("3-PHASE CURRICULUM CHESS AI TRAINING")
    print("="*80)
    print(f"Variant: {args.variant}")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Starting phase: {args.start_phase}")
    print("="*80 + "\n")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Create model
    print(f"Creating {args.variant} model...")
    model = create_model(variant=args.variant).to(device)
    
    # Print model summary
    model_summary(model)
    
    # Optional: Compile model for 2x speedup (PyTorch 2.0+)
    if HARDWARE_CONFIG['compile_model']:
        try:
            model = torch.compile(model)
            print("✓ Model compiled with torch.compile\n")
        except Exception as e:
            print(f"⚠ Could not compile model: {e}\n")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        model = load_model_with_compatibility(model, args.resume, device)
        print("✓ Checkpoint loaded\n")
    
    # Run phases
    if args.start_phase <= 1:
        model = phase1_puzzle_bootcamp(model, args.variant, args.checkpoint_dir, args)
    
    if args.start_phase <= 2:
        model = phase2_transition(model, args.variant, args.checkpoint_dir, args)
    
    if args.start_phase <= 3:
        model = phase3_pure_selfplay(model, args.variant, args.checkpoint_dir, args)
    
    print("\n✓ All phases complete!")


if __name__ == "__main__":
    mp.freeze_support()
    main()
