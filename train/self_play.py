import torch
import torch.nn.functional as F
import chess
import chess.pgn
import numpy as np
import time
import math
import psutil
import os
import random
import multiprocessing

from data import board_to_tensor, get_move_index, SelfPlayDataset
from utils import clear_memory, test_tactical_recognition

# Add import for MCTS functionality
from mcts import generate_mcts_game, select_move_with_mcts, MCTS_CONFIG

from utils import get_optimal_batch_size, clear_memory


# ============================================================================
# SELF-PLAY CONFIGURATION
# ============================================================================
SELF_PLAY_CONFIG = {
    # MCTS settings for self-play
    'num_simulations': 200,       # Simulations per move (was 10, now much higher)
    'fast_simulations': 50,       # Simulations for fast mode
    
    # Temperature schedule
    'temp_initial': 1.0,          # Temperature for first N moves
    'temp_mid': 0.5,              # Temperature for mid-game
    'temp_final': 0.1,            # Temperature for late game
    'temp_move_threshold_1': 15,  # Moves before reducing temp
    'temp_move_threshold_2': 30,  # Moves before final temp
    
    # Exploration
    'dirichlet_alpha': 0.3,       # Dirichlet noise alpha
    'dirichlet_weight': 0.25,     # Weight of noise at root
    
    # Game limits
    'max_moves': 200,             # Maximum moves per game
    'min_game_length': 10,        # Minimum game length to use
    
    # Reward shaping
    'use_reward_shaping': True,   # Apply reward shaping
    'discount_factor': 0.99,      # Gamma for reward discounting
}


def generate_reinforcement_learning_samples(model, device, num_games=100, reward_shaping=True, 
                                           iteration=0, total_iterations=5):
    """Generate self-play games with reinforcement learning objectives.
    
    Uses MCTS for move selection with proper exploration via Dirichlet noise.
    
    Args:
        model: The neural network model
        device: The computation device (CPU or GPU)
        num_games: Number of self-play games to generate
        reward_shaping: Whether to enhance rewards for checkmate/near-checkmate positions
        iteration: Current iteration number (for adjusting exploration parameters)
        total_iterations: Total number of iterations planned
        
    Returns:
        List of (board_tensor, policy_target, value_target) tuples for training
    """
    samples = []
    model.eval()
    
    # Determine input channels from model
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 22
    
    # Parallel game generation setup
    batch_size = min(16, num_games)
    active_games = [chess.Board() for _ in range(batch_size)]
    move_histories = [[] for _ in range(batch_size)]
    board_histories = [[] for _ in range(batch_size)]
    policy_histories = [[] for _ in range(batch_size)]  # Store MCTS policies
    move_numbers = [1 for _ in range(batch_size)]
    active_mask = [True for _ in range(batch_size)]
    completed_games = 0
    
    # Progressive exploration reduction
    progress_factor = iteration / max(1, total_iterations - 1)
    
    # MCTS parameters - increase simulations as training progresses for quality
    base_simulations = SELF_PLAY_CONFIG['num_simulations']
    num_simulations = max(50, int(base_simulations * (0.5 + 0.5 * progress_factor)))
    
    # Temperature scheduling
    base_temp = SELF_PLAY_CONFIG['temp_initial']
    final_temp = SELF_PLAY_CONFIG['temp_final']
    
    # Reduce noise as training progresses
    dirichlet_alpha = SELF_PLAY_CONFIG['dirichlet_alpha'] * (1 - 0.5 * progress_factor)
    dirichlet_weight = SELF_PLAY_CONFIG['dirichlet_weight'] * (1 - 0.5 * progress_factor)
    
    max_moves_per_game = SELF_PLAY_CONFIG['max_moves']
    moves_played = [0 for _ in range(batch_size)]
    
    print(f"Generating {num_games} self-play games with MCTS ({num_simulations} sims, temp={base_temp:.2f})")
    
    # Dynamic batch sizing
    current_batch_size = 4
    max_batch_size = 16
    batch_success_count = 0
    
    while any(active_mask) and completed_games < num_games:
        active_indices = [i for i, active in enumerate(active_mask) if active]
        
        if not active_indices:
            break
        
            
        # Process in dynamic batches
        try:
            # Try current batch size
            input_tensors = torch.stack([
                torch.tensor(board_to_tensor(active_games[i], move_numbers[i], input_channels), dtype=torch.float32)
                for i in active_indices[:current_batch_size]
            ]).to(device)
            
            with torch.no_grad():
                policy_logits, value_preds = model(input_tensors)
                
            # Success - consider increasing batch size
            batch_success_count += 1
            if batch_success_count >= 3 and current_batch_size < max_batch_size:
                current_batch_size = min(current_batch_size * 2, max_batch_size)
                print(f"Increased batch size to {current_batch_size}")
                
        except RuntimeError as e:  # Usually OOM
            # Reduce batch size and try again
            current_batch_size = max(current_batch_size // 2, 1)
            batch_success_count = 0
            print(f"Reduced batch size to {current_batch_size} due to error")
            clear_memory()
            continue
        
        # Get initial policy and values
        initial_policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        initial_values = value_preds.squeeze(-1).cpu().numpy()
        
        # Process each active game
        for idx, i in enumerate(active_indices):
            board = active_games[i]
            legal_moves = list(board.legal_moves)
            
            # Check for game over conditions or move limit reached
            if not legal_moves or board.is_game_over() or moves_played[i] >= max_moves_per_game:
                # Save result and create training samples with appropriate rewards
                if board.is_checkmate():
                    # Checkmate is highest reward/penalty
                    result_value = 1.0 if not board.turn else -1.0
                elif board.is_stalemate() or board.is_insufficient_material():
                    # Stalemate and insufficient material are draws
                    result_value = 0.0
                elif moves_played[i] >= max_moves_per_game:
                    # Truncated games are treated as slightly negative for both sides
                    result_value = -0.1
                else:
                    # Other game terminations (50-move rule, repetition) are draws
                    result_value = 0.0
                
                # Generate training samples from this game with updated rewards
                game_samples = create_training_samples_from_game(
                    board_histories[i], 
                    move_histories[i], 
                    result_value,
                    reward_shaping
                )
                samples.extend(game_samples)
                
                # Track completed games
                completed_games += 1
                
                # Start a new game if needed
                if completed_games < num_games:
                    active_games[i] = chess.Board()
                    move_histories[i] = []
                    board_histories[i] = []
                    move_numbers[i] = 1
                    moves_played[i] = 0
                else:
                    active_mask[i] = False
                
                continue
            
            # Get initial move probabilities from policy network
            policy = initial_policies[idx]
            move_probs = np.zeros(len(legal_moves))
            
            for move_idx, move in enumerate(legal_moves):
                move_index = get_move_index(move)
                move_probs[move_idx] = policy[move_index]
            
            # Handle case of all zero probabilities
            if np.sum(move_probs) <= 1e-10:
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)
            else:
                # Normalize (ensure probabilities sum to 1)
                move_probs = move_probs / np.sum(move_probs)
            
            # Add Dirichlet noise to root node for exploration (AlphaZero style)
            if len(legal_moves) > 0:
                noise = np.random.dirichlet([dirichlet_alpha] * len(legal_moves))
                move_probs = (1 - dirichlet_weight) * move_probs + dirichlet_weight * noise
            
            # MCTS-inspired simulations - simple version that avoids full tree search
            visit_counts = np.zeros(len(legal_moves))
            q_values = np.zeros(len(legal_moves))
            
            # Run multiple simulations to improve move selection
            for _ in range(num_simulations):
                # Select move for simulation based on UCB formula
                ucb_scores = np.zeros(len(legal_moves))
                total_visits = np.sum(visit_counts) + 1e-8
                
                for move_idx in range(len(legal_moves)):
                    if visit_counts[move_idx] > 0:
                        # UCB score balances exploitation (Q-value) with exploration (log term)
                        ucb_scores[move_idx] = q_values[move_idx] + 2.0 * np.sqrt(np.log(total_visits) / visit_counts[move_idx]) * move_probs[move_idx]
                    else:
                        # For unvisited nodes, prioritize by prior probability
                        ucb_scores[move_idx] = 1.0 + move_probs[move_idx]
                
                # Select move with highest UCB score
                sim_move_idx = np.argmax(ucb_scores)
                sim_move = legal_moves[sim_move_idx]
                
                # Simulate this move and get a value estimate
                sim_board = board.copy()
                sim_board.push(sim_move)
                
                # For efficiency, just use the model's direct evaluation 
                sim_tensor = torch.tensor(board_to_tensor(sim_board, move_numbers[i] + 1, input_channels), dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    _, sim_value = model(sim_tensor)
                    
                # Convert value to current player's perspective
                sim_value = -float(sim_value.item())  # Negative because we're evaluating from opponent's view
                
                # Update statistics
                visit_counts[sim_move_idx] += 1
                q_values[sim_move_idx] = (q_values[sim_move_idx] * (visit_counts[sim_move_idx] - 1) + sim_value) / visit_counts[sim_move_idx]
            
            # After simulations, select move based on visit counts (not raw policy)
            # Apply temperature to visit count distribution
            if np.sum(visit_counts) > 0:
                visit_counts_temp = np.power(visit_counts, 1.0 / temperature)
                visit_policy = visit_counts_temp / np.sum(visit_counts_temp)
            else:
                visit_policy = move_probs
            
            # Store current position for later training
            board_histories[i].append(board_to_tensor(board, move_numbers[i], input_channels))
            
            # Select move - early in training, explore more. Later, be more greedy.
            exploration_threshold = 0.8 + 0.1 * (1 - progress_factor)  # Decreases from 0.9 to 0.8 over time
            
            if np.random.random() < exploration_threshold:  # Mostly select best move
                selected_idx = np.argmax(visit_policy)
                move = legal_moves[selected_idx]
            else:  # Sometimes explore other moves
                selected_idx = np.random.choice(len(legal_moves), p=visit_policy)
                move = legal_moves[selected_idx]
            
            # Store selected move
            move_histories[i].append(get_move_index(move))
            
            # Make the move
            board.push(move)
            moves_played[i] += 1
            move_numbers[i] += 1
        
        # Show progress
        if completed_games > 0 and completed_games % 10 == 0:
            print(f"Completed {completed_games}/{num_games} self-play games")
    
    print(f"Generated {len(samples)} training samples from {completed_games} games")
    return samples


def create_training_samples_from_game(board_history, move_history, final_result, reward_shaping=True):
    """Create training samples from a completed self-play game
    
    This function applies reward shaping to emphasize learning from checkmate sequences
    """
    samples = []
    game_length = len(move_history)
    
    # Skip very short games
    if game_length < 5:
        return []
        
    for i in range(game_length):
        # The board state
        board_tensor = board_history[i]
        
        # The move that was actually played
        move_idx = move_history[i]
        
        # Calculate shaped reward based on position in game 
        if reward_shaping:
            # Positions closer to the end get rewards closer to the final result
            # This creates a smoother reward gradient for learning
            progress_factor = i / game_length
            
            if final_result > 0:  # Winning position
                # Reward increases exponentially toward the end
                shaped_value = final_result * min(1.0, progress_factor * 2)
            elif final_result < 0:  # Losing position
                # Penalty increases toward the end
                shaped_value = final_result * min(1.0, progress_factor * 2)
            else:  # Draw
                shaped_value = final_result * progress_factor
        else:
            # Without reward shaping, all positions get the game's final result
            shaped_value = final_result
            
        # Flip value target for black's perspective
        is_white_to_move = np.sum(board_tensor[17]) > 0  # Check the turn channel
        if not is_white_to_move:
            shaped_value = -shaped_value
        
        samples.append((board_tensor, move_idx, shaped_value))
    
    return samples


def generate_self_play_games(model, device, num_games=100, use_mcts=True):
    """Generate self-play games without reinforcement learning, now using MCTS by default"""
    games = []
    model.eval()
    
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 20

    if use_mcts:
        # Use MCTS for higher quality games (but slower generation)
        for i in range(num_games):
            if i % 5 == 0:
                print(f"Generating MCTS game {i+1}/{num_games}")
            game = generate_mcts_game(model, device, temperature=1.0, 
                                    num_simulations=100, c_puct=1.0, 
                                    parallel_workers=4, input_channels=input_channels)
            games.append(game)
        return games
    
    # Pre-allocate tensor memory
    batch_size = min(16, num_games)  # Process up to 16 games in parallel
    active_games = [chess.Board() for _ in range(batch_size)]
    game_nodes = [chess.pgn.Game() for _ in range(batch_size)]
    current_nodes = [game for game in game_nodes]
    move_numbers = [1 for _ in range(batch_size)]
    active_mask = [True for _ in range(batch_size)]
    
    # Set headers
    for game in game_nodes:
        game.headers["Result"] = "*"
    
    while any(active_mask):
        # Get all active boards that aren't in terminal state
        active_indices = [i for i, active in enumerate(active_mask) if active]
        
        if not active_indices:
            break
            
        # Batch process all active boards
        input_tensors = torch.stack([
            torch.tensor(board_to_tensor(active_games[i], move_numbers[i], input_channels=input_channels), dtype=torch.float32)
            for i in active_indices
        ]).to(device)
        
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                policy_logits, _ = model(input_tensors)
        
        policies = F.softmax(policy_logits, dim=1).cpu().numpy()
        
        # Process each active game
        for idx, i in enumerate(active_indices):
            board = active_games[i]
            legal_moves = list(board.legal_moves)
            
            if not legal_moves or board.is_game_over():
                # Game is over
                result = board.result()
                game_nodes[i].headers["Result"] = result
                games.append(game_nodes[i])
                
                # If we still need more games, start a new one
                if len(games) < num_games:
                    active_games[i] = chess.Board()
                    game_nodes[i] = chess.pgn.Game()
                    game_nodes[i].headers["Result"] = "*"
                    current_nodes[i] = game_nodes[i]
                    move_numbers[i] = 1
                else:
                    active_mask[i] = False
                continue
            
            # Get move probabilities
            policy = policies[idx]
            move_probs = np.zeros(len(legal_moves))
            
            for move_idx, move in enumerate(legal_moves):
                move_index = get_move_index(move)
                move_probs[move_idx] = policy[move_index]
            
            # Fix: Ensure we don't have all zeros by adding a small constant
            # and handle zero sums properly
            if np.sum(move_probs) <= 1e-10:
                # If all moves have essentially zero probability, use uniform distribution
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)
            else:
                # Normalize and handle potential division by zero
                move_probs = move_probs / np.sum(move_probs)
            
            # Select move
            move = np.random.choice(legal_moves, p=move_probs)
            
            # Apply move
            board.push(move)
            current_nodes[i] = current_nodes[i].add_variation(move)
            move_numbers[i] += 1
    
    # Add any remaining active games
    for i, active in enumerate(active_mask):
        if active:
            result = active_games[i].result()
            game_nodes[i].headers["Result"] = result
            games.append(game_nodes[i])
    
    return games[:num_games]  # Ensure we only return the requested number


def run_self_play_training(model, device, save_path, state_file, puzzle_dataloader=None, 
                         num_games=50, num_iterations=5, use_mcts=True, fast_mcts=False):
    """Run self-play training to improve the model through reinforcement learning.
    
    This is the main self-play training loop that:
    1. Generates self-play games using MCTS
    2. Trains on the generated positions
    3. Interleaves puzzle training for tactical awareness
    4. Evaluates progress with tactical benchmarks
    
    Args:
        model: The neural network
        device: Computation device
        save_path: Path to save model checkpoints
        state_file: Path to save training state
        puzzle_dataloader: DataLoader for tactical puzzles
        num_games: Number of games per iteration
        num_iterations: Number of training iterations
        use_mcts: Whether to use MCTS for game generation
        fast_mcts: Use faster MCTS settings
    """
    print(f"\n{'='*60}")
    print("SELF-PLAY REINFORCEMENT LEARNING")
    print(f"{'='*60}")
    print(f"Iterations: {num_iterations}, Games per iteration: {num_games}")
    print(f"MCTS: {'Fast' if fast_mcts else 'Full' if use_mcts else 'Disabled'}")
    
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 22
    
    # Use SGD with momentum (AlphaZero-style) for better convergence
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=0.01,  # Will be scheduled
        momentum=0.9, 
        weight_decay=1e-4,
        nesterov=True
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iterations, eta_min=1e-5
    )
    
    # Loss functions with improvements
    from training import PolicyLoss, ValueLoss
    policy_loss_fn = PolicyLoss()
    value_loss_fn = ValueLoss(use_huber=True)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Find optimal batch size
    batch_size = get_optimal_batch_size(model, device, starting_size=32, min_size=8)
    print(f"Batch size: {batch_size}")
    
    # Track progress
    total_positions = 0
    best_accuracy = 0
    
    # MCTS settings based on mode
    if fast_mcts:
        mcts_simulations = SELF_PLAY_CONFIG['fast_simulations']
    else:
        mcts_simulations = SELF_PLAY_CONFIG['num_simulations']
    
    # Gradient clipping threshold
    grad_clip = 1.0
    
    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration+1}/{num_iterations} ---")
        
        # Progressive training adjustments
        progress_factor = iteration / max(1, num_iterations - 1)
        
        # Increase value weight over time (initially focus on policy)
        policy_weight = 1.0
        value_weight = 1.0 + progress_factor * 1.0  # 1.0 -> 2.0
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"LR: {current_lr:.6f}, Policy weight: {policy_weight:.2f}, Value weight: {value_weight:.2f}")
        
        # Monitor memory usage before generation
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage before game generation: {mem_before:.1f} MB")
        
        # Generate games using MCTS with conservative settings
        if use_mcts:
            # Generate games using MCTS if enabled
            print(f"Generating {num_games} self-play games using MCTS...")
            
            # Adjust parameters based on mode
            if fast_mcts:
                # Fast MCTS settings - use simplified MCTS algorithm
                print("Using simplified MCTS for faster training")
                num_simulations = 50
                adjusted_games = max(10, min(20, num_games // 5))
                
                games = []
                # Generate games one at a time with memory cleanup between each
                for i in range(adjusted_games):
                    print(f"Generating simple MCTS game {i+1}/{adjusted_games}")
                    try:
                        # Use simplified MCTS with moderate simulation count
                        game = generate_simple_mcts_game(
                            model, 
                            device, 
                            temperature=1.0,
                            num_simulations=num_simulations
                        )
                        games.append(game)
                        
                        # Force cleanup between games
                        if i % 2 == 1:  # Every other game
                            clear_memory()
                    except Exception as e:
                        print(f"Error generating game: {e}")
                        clear_memory()
                        continue
            else:
                # Full MCTS settings - more thorough but slower
                num_simulations = 200
                parallel_workers = 5
                adjusted_games = max(5, min(10, num_games // 10))
                
                games = []
                
                # Generate games one at a time with memory cleanup between each
                for i in range(adjusted_games):
                    print(f"Generating MCTS game {i+1}/{adjusted_games}")
                    try:
                        # Use full MCTS with higher simulation count
                        game = generate_mcts_game(
                            model, 
                            device, 
                            temperature=1.0,
                            num_simulations=num_simulations,
                            c_puct=1.0,
                            parallel_workers=parallel_workers,
                            input_channels=input_channels
                        )
                        games.append(game)
                        
                        # Force cleanup between games
                        if i % 2 == 1:  # Every other game
                            clear_memory()
                    except Exception as e:
                        print(f"Error generating game: {e}")
                        clear_memory()
                        continue
            
            # Convert games to training samples with memory management
            if games:
                self_play_samples = []
                for game in games:
                    board = chess.Board()
                    result_str = game.headers.get("Result", "*")
                    result_value = 0.0  # Default for unfinished games
                    if result_str == "1-0":
                        result_value = 1.0
                    elif result_str == "0-1":
                        result_value = -1.0
                    
                    move_history = []
                    board_history = []
                    move_number = 1
                    
                    for move in game.mainline_moves():
                        board_history.append(board_to_tensor(board, move_number, input_channels))
                        move_history.append(get_move_index(move))
                        board.push(move)
                        move_number += 1
                    
                    game_samples = create_training_samples_from_game(
                        board_history, 
                        move_history, 
                        result_value,
                        True  # Always use reward shaping for MCTS games
                    )
                    self_play_samples.extend(game_samples)
                    
                    # Clear references to help with memory
                    del board_history
                    del move_history
            else:
                print("Failed to generate valid self-play games with MCTS. Falling back to non-MCTS.")
                # Fall back to non-MCTS
                self_play_samples = generate_reinforcement_learning_samples(
                    model,
                    device, 
                    num_games=max(10, num_games // 5),  # Generate fewer games as a fallback
                    reward_shaping=True,
                    iteration=iteration,
                    total_iterations=num_iterations
                )
        else:
            # Phase 1: Generate self-play games with reward shaping for checkmate
            # Use fewer games to avoid crashes
            adjusted_games = max(10, num_games // 5)  # Generate fewer games
            print(f"Generating {adjusted_games} self-play games...")
            self_play_samples = generate_reinforcement_learning_samples(
                model,
                device, 
                num_games=adjusted_games,
                reward_shaping=True,
                iteration=iteration,
                total_iterations=num_iterations
            )
        
        # Check memory after generation
        mem_after = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage after game generation: {mem_after:.1f} MB (change: {mem_after-mem_before:.1f} MB)")
        
        # Force cleanup
        clear_memory()
        
        if not self_play_samples or len(self_play_samples) < 100:
            print("Failed to generate enough valid self-play samples. Skipping iteration.")
            continue
            
            
        print(f"Generated {len(self_play_samples)} training positions from self-play")
        
        # Phase 2: Train on the generated positions with smaller batch size and more frequent cleanup
        print("Training on self-play positions with integrated tactical training...")
        rl_dataset = SelfPlayDataset(self_play_samples, "small" if model.is_small_model() else "big")
        rl_dataloader = torch.utils.data.DataLoader(
            rl_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4, 
            pin_memory=True
        )
        
        # Training loop for this iteration
        model.train()
        total_rl_loss = 0
        batch_count = 0
        
        # Determine if we can train on puzzles
        can_train_puzzles = puzzle_dataloader is not None
        puzzle_policy_weight = 3.0
        puzzle_value_weight = 2.0
        puzzle_batches_per_rl_batch = 10  # Train on 10 puzzle batches after each RL batch
        num_epochs = 5

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            total_rl_loss = 0
            batch_count = 0
    
        
            for batch in rl_dataloader:
                # Train on self-play batch
                inputs, policy_targets, value_targets = batch
                inputs = inputs.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                
                optimizer.zero_grad()
                
                # Regular self-play training code...
                if scaler:
                    with torch.amp.autocast(device_type="cuda"):
                        policy_logits, value_pred = model(inputs)
                        policy_loss = policy_loss_fn(policy_logits, policy_targets)
                        value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                        loss = policy_weight * policy_loss + value_weight * value_loss
                    
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
                
                total_rl_loss += loss.item()
                batch_count += 1
                
                if batch_count % 5 == 0:
                    print(f"  Self-play batch {batch_count}, Loss: {loss.item():.4f}")
                
                # Now train on puzzle batches after each self-play batch
                if can_train_puzzles:
                    print(f"  Training on puzzle batches after self-play batch {batch_count}...")
                    puzzle_losses = []
                    
                    # Train on multiple puzzle batches for each self-play batch
                    for _ in range(puzzle_batches_per_rl_batch):
                        # Get a batch of puzzles
                        puzzle_batch = next(iter(puzzle_dataloader))
                        
                        # Handle both 3-element and 4-element batches
                        if len(puzzle_batch) == 3:
                            p_inputs, p_policy_targets, p_value_targets = puzzle_batch
                            categories = ["default"] * p_inputs.size(0)
                        else:
                            p_inputs, p_policy_targets, p_value_targets, categories = puzzle_batch
                        
                        p_inputs = p_inputs.to(device)
                        p_policy_targets = p_policy_targets.to(device)
                        p_value_targets = p_value_targets.to(device)
                        
                        optimizer.zero_grad()
                        
                        # Training with category-specific weighting
                        if scaler:
                            with torch.amp.autocast(device_type="cuda"):
                                p_policy_logits, p_value_pred = model(p_inputs)
                                
                                # Apply category weights
                                puzzle_loss = 0
                                for i, category in enumerate(categories):
                                    # Prioritize checkmates and endgames
                                    if category == "mate_in_one":
                                        cat_weight = 5.0
                                    elif category == "endgame":
                                        cat_weight = 3.0
                                    elif category in ["knight_fork", "pin", "discovered"]:
                                        cat_weight = 2.0
                                    else:
                                        cat_weight = 1.0
                                    
                                    p_loss = policy_loss_fn(p_policy_logits[i:i+1], p_policy_targets[i:i+1])
                                    v_loss = value_loss_fn(p_value_pred[i:i+1].squeeze(), p_value_targets[i:i+1])
                                    sample_loss = (puzzle_policy_weight * p_loss + puzzle_value_weight * v_loss) * cat_weight
                                    puzzle_loss += sample_loss
                                
                            scaler.scale(puzzle_loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            # Non-scaler version of puzzle training
                            p_policy_logits, p_value_pred = model(p_inputs)
                            puzzle_loss = 0
                            
                            for i, category in enumerate(categories):
                                # Prioritize checkmates and endgames
                                if category == "mate_in_one":
                                    cat_weight = 5.0
                                elif category == "endgame":
                                    cat_weight = 3.0
                                elif category in ["knight_fork", "pin", "discovered"]:
                                    cat_weight = 2.0
                                else:
                                    cat_weight = 1.0
                                
                                p_loss = policy_loss_fn(p_policy_logits[i:i+1], p_policy_targets[i:i+1])
                                v_loss = value_loss_fn(p_value_pred[i:i+1].squeeze(), p_value_targets[i:i+1])
                                sample_loss = (puzzle_policy_weight * p_loss + puzzle_value_weight * v_loss) * cat_weight
                                puzzle_loss += sample_loss
                            
                            puzzle_loss.backward()
                            optimizer.step()
                        
                        puzzle_losses.append(puzzle_loss.item())
                    
                    if puzzle_losses:
                        avg_puzzle_loss = sum(puzzle_losses) / len(puzzle_losses)
                        print(f"    Puzzle batch loss: {avg_puzzle_loss:.4f}")
                
                # Memory cleanup every few batches
                if batch_count % 10 == 0:
                    clear_memory()
            
            if batch_count > 0:
                avg_loss = total_rl_loss / batch_count
                print(f"Avg training loss: {avg_loss:.4f}")
            
            # Save checkpoint after each iteration
            torch.save(model.state_dict(), save_path)
            print(f"Model checkpoint saved after iteration {iteration+1}")
            
            # Test tactical recognition after each iteration
            print("Testing tactical recognition...")
            test_accuracy = test_tactical_recognition(model, device)
            print(f"Tactical recognition accuracy: {test_accuracy:.2%}")
            
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                # Save best model separately
                torch.save(model.state_dict(), save_path.replace('.pth', '_best.pth'))
                print(f"New best model saved with accuracy: {best_accuracy:.2%}")
            total_positions += len(self_play_samples)
            
            # Clean up between iterations
            del self_play_samples
            del rl_dataset
            del rl_dataloader
            clear_memory()
            
            # Add a brief pause to let system cool down
            print("Pausing for 5 seconds to allow system recovery...")
            time.sleep(5)
            
            # ADD THIS SECTION: Integrated tactical training after each self-play iteration
            if puzzle_dataloader is not None:
                print("\n=== INTEGRATED TACTICAL TRAINING PHASE ===")
                
                # Create optimizer specifically for tactics (SGD for consistency)
                tactical_optimizer = torch.optim.SGD(model.parameters(), lr=0.0008, momentum=0.9, weight_decay=1e-4)
                
                # Calculate dynamic epochs based on progress - more later in training
                tactical_epochs = 2 + int(iteration / num_iterations * 3)  # 2-5 epochs
                print(f"Running tactical training for {tactical_epochs} epochs")
                
                # Weight tactical positions even higher than regular ones
                puzzle_policy_weight = 3.0
                puzzle_value_weight = 2.0
                
                # Tactical training loop
                model.train()
                total_loss = 0
                batch_count = 0
                
                for epoch in range(tactical_epochs):
                    for batch in puzzle_dataloader:
                        # Handle both 3-element and 4-element batches
                        if len(batch) == 3:
                            inputs, policy_targets, value_targets = batch
                            categories = ["default"] * inputs.size(0)
                        else:
                            inputs, policy_targets, value_targets, categories = batch
                        
                        inputs = inputs.to(device)
                        policy_targets = policy_targets.to(device)
                        value_targets = value_targets.to(device)
                        
                        tactical_optimizer.zero_grad()
                        
                        # Training with category-specific weighting
                        if scaler:
                            with torch.amp.autocast(device_type="cuda"):
                                policy_logits, value_pred = model(inputs)
                                
                                # Weight different tactical positions differently
                                loss = 0
                                for i, category in enumerate(categories):
                                    # Get single sample
                                    policy_logit = policy_logits[i:i+1]
                                    policy_target = policy_targets[i:i+1]
                                    value_target = value_targets[i:i+1]
                                    value_p = value_pred[i:i+1]
                                    
                                    # Prioritize checkmates and endgames
                                    if category == "mate_in_one":
                                        cat_weight = 5.0
                                    elif category == "endgame":
                                        cat_weight = 3.0
                                    elif category in ["knight_fork", "pin", "discovered"]:
                                        cat_weight = 2.0
                                    else:
                                        cat_weight = 1.0
                                    
                                    p_loss = policy_loss_fn(policy_logit, policy_target)
                                    v_loss = value_loss_fn(value_p.squeeze(), value_target)
                                    sample_loss = (puzzle_policy_weight * p_loss + puzzle_value_weight * v_loss) * cat_weight
                                    loss += sample_loss
                                
                            scaler.scale(loss).backward()
                            scaler.step(tactical_optimizer)
                            scaler.update()
                        else:
                            # Similar non-scaler code
                            policy_logits, value_pred = model(inputs)
                            loss = 0
                            
                            for i, category in enumerate(categories):
                                # Get single sample
                                policy_logit = policy_logits[i:i+1]
                                policy_target = policy_targets[i:i+1]
                                value_target = value_targets[i:i+1]
                                value_p = value_pred[i:i+1]
                                
                                # Prioritize checkmates and endgames
                                if category == "mate_in_one":
                                    cat_weight = 5.0
                                elif category == "endgame":
                                    cat_weight = 4.0
                                elif category in ["knight_fork", "pin", "discovered"]:
                                    cat_weight = 3.0
                                else:
                                    cat_weight = 1.0
                                
                                p_loss = policy_loss_fn(policy_logit, policy_target)
                                v_loss = value_loss_fn(value_p.squeeze(), value_target)
                                sample_loss = (puzzle_policy_weight * p_loss + puzzle_value_weight * v_loss) * cat_weight
                                loss += sample_loss
                            
                            loss.backward()
                            tactical_optimizer.step()
                        
                        total_loss += loss.item()
                        batch_count += 1
                        
                        # Process fewer batches per epoch to save time
                        if batch_count % epoch == 5:
                            break
                    
                    print(f"Tactical epoch {epoch+1}/{tactical_epochs}, loss: {loss.item():.4f}")
                    
                    # Memory cleanup after each tactical epoch
                    clear_memory()

                if batch_count > 0:
                    print(f"Average tactical loss: {total_loss/batch_count:.4f}")
        
    
    print(f"\n=== SELF-PLAY TRAINING COMPLETED ===")
    print(f"Processed {total_positions} positions across {num_iterations} iterations")
    print(f"Best tactical accuracy: {best_accuracy:.2%}")
    return model



class SimpleNode:
    def __init__(self, board, move=None, parent=None, prior=0):
        self.board = board
        self.move = move
        self.parent = parent
        self.prior = prior
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
    
    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

def simple_mcts_for_training(board, model, device, num_simulations=50, temperature=1.0, depth_limit=6):
    """Enhanced MCTS with recursive search and backpropagation"""
    # Create root node
    root = SimpleNode(board)
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 20
    # Initial policy from model
    input_tensor = torch.tensor(board_to_tensor(board, board.fullmove_number, input_channels), 
                              dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(input_tensor)
    policy = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()
    
    # Expand root with all legal moves
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        move_index = get_move_index(move)
        prior = policy[move_index] if move_index < len(policy) else 0.001
        next_board = board.copy()
        next_board.push(move)
        root.children[move] = SimpleNode(next_board, move, root, prior)
    
    # Add Dirichlet noise at root
    if legal_moves:
        noise = np.random.dirichlet([0.3] * len(legal_moves))
        for i, move in enumerate(legal_moves):
            root.children[move].prior = 0.75 * root.children[move].prior + 0.25 * noise[i]
    
    # Run simulations
    for _ in range(num_simulations):
        # Selection and expansion phase - traverse tree to leaf
        node, depth = root, 0
        while node.children and depth < depth_limit:
            # Select child with highest UCB
            best_score = -float('inf')
            best_move = None
            
            for move, child in node.children.items():
                # UCB formula
                if child.visit_count == 0:
                    score = 1000 + child.prior  # Prioritize unexplored
                else:
                    exploit = child.value()
                    explore = 2.0 * child.prior * (np.sqrt(node.visit_count) / (1 + child.visit_count))
                    score = exploit + explore
                
                if score > best_score:
                    best_score = score
                    best_move = move
            
            # Move down the tree
            node = node.children[best_move]
            depth += 1
        
        # Expand if node is not terminal and we're not at depth limit
        if not node.board.is_game_over() and depth < depth_limit and not node.children:
            # Get policy from model
            input_channels = model.input_channels if hasattr(model, 'input_channels') else 20
            input_tensor = torch.tensor(board_to_tensor(node.board, node.board.fullmove_number, input_channels), 
                                     dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, _ = model(input_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()
            
            # Create children
            for move in node.board.legal_moves:
                move_index = get_move_index(move)
                prior = policy[move_index] if move_index < len(policy) else 0.001
                next_board = node.board.copy()
                next_board.push(move)
                node.children[move] = SimpleNode(next_board, move, node, prior)
        
        # Evaluate position
        if node.board.is_game_over():
            # Terminal position - real game result
            if node.board.is_checkmate():
                value = -1.0  # -1 because it's from the perspective of the player who just moved
            else:
                value = 0.0  # Draw
        else:
            # Use neural network for non-terminal positions
            input_tensor = torch.tensor(board_to_tensor(node.board, node.board.fullmove_number, input_channels), 
                                     dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _, value_tensor = model(input_tensor)
            value = float(value_tensor.item())
        
        # Backpropagate
        while node:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent's perspective
            node = node.parent
    
    # Select move based on visit counts
    visits = np.array([root.children[move].visit_count for move in legal_moves])
    
    # Apply temperature
    if temperature == 0:  # Deterministic
        best_idx = np.argmax(visits)
        probs = np.zeros_like(visits)
        probs[best_idx] = 1.0
    else:
        # Apply temperature
        visits_temp = np.power(visits, 1.0 / temperature)
        if np.sum(visits_temp) > 0:
            probs = visits_temp / np.sum(visits_temp)
        else:
            probs = np.ones(len(legal_moves)) / len(legal_moves)
    
    # Return both selected move and full distribution for training
    selected_idx = np.random.choice(len(legal_moves), p=probs)
    selected_move = legal_moves[selected_idx]
    
    return selected_move, probs


def generate_simple_mcts_game(model, device, temperature=1.0, num_simulations=50):
    """Generate a self-play game using the simplified MCTS (faster for training)"""
    # Get half the available CPU cores
    parallel_workers = math.ceil(max(1, multiprocessing.cpu_count() // 1.5))
    print(f"Using {parallel_workers} CPU cores for simple MCTS")
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 20
    
    game = chess.pgn.Game()
    board = chess.Board()
    node = game
    move_number = 1
    
    # Apply early termination for very long games
    max_moves = 80  # Limit to reasonable game length
    
    while not board.is_game_over() and move_number <= max_moves:
        # Temperature annealing - reduce temperature as game progresses  
        if board.fullmove_number < 10:
            current_temp = temperature
        elif board.fullmove_number < 30:
            current_temp = temperature * 0.75
        else:
            current_temp = temperature * 0.5
            
        # Select move using simplified MCTS
        try:
            move, _ = simple_mcts_for_training(
                board, 
                model, 
                device,
                num_simulations=num_simulations,  # Default is now 100
                temperature=current_temp
            )
        except Exception as e:
            print(f"MCTS error: {e}. Falling back to direct move selection.")
            # Fallback to direct move selection if MCTS fails
            input_tensor = torch.tensor(board_to_tensor(board, move_number, input_channels), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, _ = model(input_tensor)
            policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
            
            legal_moves = list(board.legal_moves)
            move_probs = np.zeros(len(legal_moves))
            
            for move_idx, move in enumerate(legal_moves):
                move_index = get_move_index(move)
                if move_index < len(policy):
                    move_probs[move_idx] = policy[move_index]
                    
            if np.sum(move_probs) <= 1e-10:
                move_probs = np.ones(len(legal_moves)) / len(legal_moves)
            else:
                move_probs = move_probs / np.sum(move_probs)
                
            move = np.random.choice(legal_moves, p=move_probs)
            
        if move is None:
            break
            
        # Add move to game
        board.push(move)
        node = node.add_variation(move)
        move_number += 1
        
        # Periodic memory cleanup during game generation
        if move_number % 20 == 0:
            clear_memory()
    
    # Set result header based on game outcome
    if board.is_checkmate():
        game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_stalemate() or board.is_insufficient_material():
        game.headers["Result"] = "1/2-1/2"
    elif board.is_fifty_moves() or board.is_repetition(3) or move_number > max_moves:
        game.headers["Result"] = "1/2-1/2"
    else:
        game.headers["Result"] = "*"  # Unfinished
        
    return game