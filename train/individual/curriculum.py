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
import chess.pgn
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ThreadPoolExecutor

# Import from core
from core.models import create_model
from core.data import (
    PuzzleDataset,
    ChessDataset,
    load_lichess_puzzles,
    load_training_examples_from_chess_pgns,
    create_balanced_concat_dataloader,
)
from core.constants import (
    MODEL_CONFIG, CURRICULUM_CONFIG, TRAINING_CONFIG, HARDWARE_CONFIG
)
from core.utils import clear_memory, test_tactical_recognition

# Import checkmate training from this module
from .checkmate import run_checkmate_bootcamp, run_checkmate_reinforcement


def _get_device():
    """Get the computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _variant_model_type(variant: str) -> str:
    input_channels = MODEL_CONFIG[variant]["input_channels"]
    if input_channels >= 22:
        return "big"
    if input_channels >= 20:
        return "medium"
    return "small"


def _chess_pgn_root() -> str:
    return str(Path(__file__).resolve().parents[1] / "chess_pgns")


def _create_dataloader(dataset, batch_size, shuffle=True, num_workers=None):
    """Create a DataLoader with sensible defaults."""
    device = _get_device()
    if num_workers is None:
        num_workers = int(HARDWARE_CONFIG.get('dataloader_workers', 4))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == 'cuda',
        persistent_workers=(num_workers > 0),
        prefetch_factor=int(HARDWARE_CONFIG.get('prefetch_factor', 2)) if num_workers > 0 else None,
        drop_last=shuffle,
    )


def _amp_enabled(device: torch.device) -> bool:
    return bool(device.type == 'cuda' and HARDWARE_CONFIG.get('enable_amp', False))


def _save_games_pgn(games, path: str) -> None:
    """Save a list of chess.pgn.Game objects to a PGN file."""
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    with open(path, "w", encoding="utf-8") as f:
        for game in games:
            try:
                f.write(game.accept(exporter))
                f.write("\n\n")
            except Exception:
                # Skip any malformed game object
                continue


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
    
    # Load puzzles from local PGN corpus first, then fall back to Lichess cache
    print("Loading puzzle dataset from train/chess_pgns/puzzles (or Lichess fallback)...")
    bundle = load_training_examples_from_chess_pgns(
        root_dir=_chess_pgn_root(),
        include_games=False,
        include_puzzles=True,
    )
    puzzles = bundle["puzzles"] or load_lichess_puzzles()
    print(f"✓ Loaded {len(puzzles)} puzzles\n")
    
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles,
        model_type=_variant_model_type(variant),
        cache_dir=os.path.join(checkpoint_dir, "cache"),
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

    use_amp = _amp_enabled(device)
    scaler = GradScaler(enabled=use_amp)
    
    best_accuracy = 0.0
    
    for epoch in range(CURRICULUM_CONFIG['phase1_epochs']):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(puzzle_loader):
            states, policy_targets, value_targets = batch[:3]
            states = states.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                policy_logits, value_preds = model(states)
                policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
                value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
                loss = policy_loss + value_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
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
    
    # Save games (PGN)
    replay_path = os.path.join(replay_dir, "phase2_games.pgn")
    _save_games_pgn(games, replay_path)
    
    if not games:
        print("Skipping training (no games generated)")
        return model
    
    # Create datasets
    selfplay_dataset = ChessDataset(
        games=games,
        augment=True,
        model_type=_variant_model_type(variant),
    )

    # Optional supervised dataset from local pro/high-elo PGNs.
    supervised_bundle = load_training_examples_from_chess_pgns(
        root_dir=_chess_pgn_root(),
        include_games=True,
        include_puzzles=False,
        game_subdirs=("pros", "high_elo"),
    )
    supervised_games = supervised_bundle["games"]
    supervised_dataset = None
    if supervised_games:
        supervised_dataset = ChessDataset(
            games=supervised_games,
            augment=True,
            model_type=_variant_model_type(variant),
        )
    
    bundle = load_training_examples_from_chess_pgns(
        root_dir=_chess_pgn_root(),
        include_games=False,
        include_puzzles=True,
    )
    puzzles = bundle["puzzles"] or load_lichess_puzzles()
    puzzle_dataset = PuzzleDataset(
        puzzles=puzzles[:10000],
        model_type=_variant_model_type(variant),
        cache_dir=os.path.join(checkpoint_dir, "cache"),
    )
    
    datasets = [selfplay_dataset]
    source_weights = [1.0]
    print(f"  self-play: {len(selfplay_dataset)} samples")

    if supervised_dataset is not None:
        datasets.append(supervised_dataset)
        source_weights.append(1.0)
        print(f"  supervised: {len(supervised_dataset)} samples")

    if puzzle_dataset is not None:
        datasets.append(puzzle_dataset)
        source_weights.append(1.5)
        print(f"  puzzles: {len(puzzle_dataset)} samples")

    dataloader = create_balanced_concat_dataloader(
        datasets,
        batch_size=CURRICULUM_CONFIG['phase1_batch_size'],
        source_weights=source_weights,
        num_workers=HARDWARE_CONFIG.get('dataloader_workers', 4),
        pin_memory=device.type == 'cuda',
        persistent_workers=HARDWARE_CONFIG.get('dataloader_workers', 4) > 0,
        prefetch_factor=HARDWARE_CONFIG.get('prefetch_factor', 2),
    )

    print(f"  balanced mix: {len(dataloader.dataset)} samples total")
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    
    use_amp = _amp_enabled(device)
    scaler = GradScaler(enabled=use_amp)

    for epoch in range(CURRICULUM_CONFIG['phase2_epochs']):
        model.train()
        total_loss = 0
        
        for states, policy_targets, value_targets in dataloader:
            states = states.to(device, non_blocking=True)
            policy_targets = policy_targets.to(device, non_blocking=True)
            value_targets = value_targets.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=use_amp):
                policy_logits, value_preds = model(states)
                policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
                value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
                loss = policy_loss + value_loss

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                optimizer.step()
            
            total_loss += loss.item()
        
        if len(dataloader) == 0:
            print(f"Epoch {epoch+1}/{CURRICULUM_CONFIG['phase2_epochs']}: no samples (skipping)")
        else:
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
                         generate_games_fn=None, puzzle_dataset=None,
                         max_iterations=None, save_every: int = 1,
                         keep_last: int = 3, keep_every: int = 15,
                         resume_path: str | None = None):
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
    
    # Continue iteration numbering from existing checkpoints.
    iteration = 0
    try:
        existing = []
        for f in os.listdir(checkpoint_dir):
            i = None
            if f.startswith("phase3_iter_") and f.endswith(".pt"):
                try:
                    i = int(f[len("phase3_iter_"):-len(".pt")])
                except ValueError:
                    i = None
            if i is not None:
                existing.append(i)
        if existing:
            iteration = max(existing)
    except Exception:
        pass
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=TRAINING_CONFIG['sgd_lr'],
        momentum=TRAINING_CONFIG['sgd_momentum'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    use_amp = _amp_enabled(device)
    scaler = GradScaler(enabled=use_amp)

    # Optional: resume optimizer/scaler from a rich checkpoint.
    if resume_path:
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict):
                # Model weights might already be loaded by main; safe to apply again.
                state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
                if state_dict is not None:
                    model.load_state_dict(state_dict, strict=False)
                opt_state = ckpt.get("optimizer_state_dict")
                if opt_state is not None:
                    try:
                        optimizer.load_state_dict(opt_state)
                    except Exception:
                        pass
                sc_state = ckpt.get("scaler_state_dict")
                if use_amp and sc_state is not None:
                    try:
                        scaler.load_state_dict(sc_state)
                    except Exception:
                        pass
                ckpt_iter = ckpt.get("iteration")
                if isinstance(ckpt_iter, int) and ckpt_iter > iteration:
                    iteration = ckpt_iter
        except Exception:
            pass

    def _parse_iter_checkpoint(name: str) -> int | None:
        prefix = "phase3_iter_"
        suffix = ".pt"
        if not (name.startswith(prefix) and name.endswith(suffix)):
            return None
        num = name[len(prefix):-len(suffix)]
        try:
            return int(num)
        except ValueError:
            return None

    def _prune_phase3_checkpoints() -> None:
        try:
            files = [f for f in os.listdir(checkpoint_dir) if f.startswith("phase3_iter_") and f.endswith(".pt")]
            iters = []
            for f in files:
                i = _parse_iter_checkpoint(f)
                if i is not None:
                    iters.append((i, f))
            if not iters:
                return
            iters.sort(key=lambda x: x[0])
            all_i = [i for i, _ in iters]
            keep = set()
            if all_i:
                keep.add(all_i[0])  # keep the first iteration checkpoint
            if keep_every and keep_every > 0:
                keep |= {i for i in all_i if i % keep_every == 0}
            if keep_last and keep_last > 0:
                keep |= set(all_i[-keep_last:])
            for i, f in iters:
                if i in keep:
                    continue
                try:
                    os.remove(os.path.join(checkpoint_dir, f))
                except OSError:
                    pass
        except Exception:
            pass
    
    def _snapshot_state_dict_cpu():
        # Snapshot the *current* model weights for background self-play.
        # Must be CPU tensors so worker processes don't create CUDA contexts.
        with torch.no_grad():
            return {k: v.detach().cpu() for k, v in model.state_dict().items()}

    try:
        executor = ThreadPoolExecutor(max_workers=1) if generate_games_fn is not None else None
        next_games_future = None

        # Warmup: generate an initial batch so we have something to train on.
        if generate_games_fn is None:
            print("⚠ No game generation function, skipping Phase 3.")
            return model

        print(f"Generating {CURRICULUM_CONFIG['phase3_games_per_iteration']} games (warmup)...")
        start_time = time.time()
        games = generate_games_fn(
            model=model,
            device=device,
            num_games=CURRICULUM_CONFIG['phase3_games_per_iteration'],
            num_simulations=CURRICULUM_CONFIG['phase3_mcts_sims']
        )
        gen_time = time.time() - start_time
        print(f"✓ Generated {len(games)} games in {gen_time:.1f}s\n")

        while True:
            iteration += 1
            if max_iterations is not None and iteration > int(max_iterations):
                print("\nReached --max-iterations, stopping Phase 3.")
                break
            print(f"\n{'='*80}")
            print(f"SELF-PLAY ITERATION {iteration}")
            print(f"{'='*80}\n")

            # Start generating the NEXT batch in the background while we train on the CURRENT batch.
            if executor is not None and next_games_future is None:
                print("Launching background self-play for next iteration...")
                snapshot = _snapshot_state_dict_cpu()
                next_games_future = executor.submit(
                    generate_games_fn,
                    model_state_dict=snapshot,
                    device=device,
                    num_games=CURRICULUM_CONFIG['phase3_games_per_iteration'],
                    num_simulations=CURRICULUM_CONFIG['phase3_mcts_sims'],
                )
            
            # Save games
            replay_path = os.path.join(replay_dir, f"iteration_{iteration:04d}.pgn")
            _save_games_pgn(games, replay_path)
            
            # Train
            selfplay_dataset = ChessDataset(
                games=games,
                augment=True,
                model_type=_variant_model_type(variant),
            )
            
            dataloader = _create_dataloader(
                selfplay_dataset,
                batch_size=CURRICULUM_CONFIG['phase3_batch_size'],
                shuffle=True
            )

            if len(dataloader) == 0:
                print("⚠ No training samples from generated games (skipping training for this iteration)")
                # Let the loop continue; the next self-play batch may contain finished games.
                continue
            
            print(f"Training for {CURRICULUM_CONFIG['phase3_training_epochs']} epochs...")
            for epoch in range(CURRICULUM_CONFIG['phase3_training_epochs']):
                model.train()
                total_loss = 0
                
                for states, policy_targets, value_targets in dataloader:
                    states = states.to(device, non_blocking=True)
                    policy_targets = policy_targets.to(device, non_blocking=True)
                    value_targets = value_targets.to(device, non_blocking=True)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    with autocast(enabled=use_amp):
                        policy_logits, value_preds = model(states)
                        policy_loss = torch.nn.functional.cross_entropy(policy_logits, policy_targets)
                        value_loss = torch.nn.functional.mse_loss(value_preds.squeeze(), value_targets)
                        loss = policy_loss + value_loss

                    if use_amp:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['grad_clip'])
                        optimizer.step()
                    
                    total_loss += loss.item()
                
                if len(dataloader) == 0:
                    print(f"  Epoch {epoch+1}: no samples")
                else:
                    avg_loss = total_loss / len(dataloader)
                    print(f"  Epoch {epoch+1}: Loss={avg_loss:.4f}")
            
            # Save checkpoint
            if save_every > 0 and (iteration % save_every == 0):
                iter_path = os.path.join(checkpoint_dir, f"phase3_iter_{iteration:04d}.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict() if use_amp else None,
                        "iteration": iteration,
                        "variant": variant,
                    },
                    iter_path,
                )
                _prune_phase3_checkpoints()
                print(f"\n✓ Iteration {iteration} complete, saved to {iter_path}")
            else:
                print(f"\n✓ Iteration {iteration} complete")
            
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

            # Collect the next batch (wait only if generation is slower than training + extras)
            if next_games_future is not None:
                print("Waiting for next self-play batch...")
                games = next_games_future.result()
                next_games_future = None
                print(f"✓ Next batch ready: {len(games)} games")

        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
        final_path = os.path.join(checkpoint_dir, "phase3_interrupted.pt")
        torch.save(model.state_dict(), final_path)
        print(f"Model saved to {final_path}")
    
    return model
