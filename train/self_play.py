import torch
import torch.nn.functional as F
import chess
import chess.pgn
import numpy as np
import time
from typing import List, Tuple, Optional

from data import board_to_tensor, get_move_index, SelfPlayDataset
from utils import clear_memory, test_tactical_recognition

# Add import for MCTS functionality
from mcts import generate_mcts_game


def generate_reinforcement_learning_samples(model, device, num_games=100, reward_shaping=True, iteration=0, total_iterations=5):
    """Generate self-play games with reinforcement learning objectives and MCTS-inspired search
    
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
    
    # Parallel game generation
    batch_size = min(16, num_games)
    active_games = [chess.Board() for _ in range(batch_size)]
    move_histories = [[] for _ in range(batch_size)]
    board_histories = [[] for _ in range(batch_size)]
    move_numbers = [1 for _ in range(batch_size)]
    active_mask = [True for _ in range(batch_size)]
    completed_games = 0
    
    # MCTS simulation parameters
    num_simulations = 10  # Number of simulations per move - keep relatively low for speed
    
    # Adaptive temperature - start high for exploration, decrease for exploitation
    progress_factor = iteration / max(1, total_iterations - 1)
    base_temp = 1.0
    final_temp = 0.5
    temperature = base_temp - progress_factor * (base_temp - final_temp)
    
    # Dirichlet noise parameters - higher alpha = more uniform noise
    dirichlet_alpha = 0.3 * (1 - 0.5 * progress_factor)  # Decrease from 0.3 to 0.15 as training progresses
    dirichlet_weight = 0.25 * (1 - 0.5 * progress_factor)  # Decrease from 0.25 to 0.125 as training progresses
    
    # Some games may go very long - limit total moves to avoid infinite games
    max_moves_per_game = 200
    moves_played = [0 for _ in range(batch_size)]
    
    print(f"Generating self-play games with MCTS (temp={temperature:.2f}, noise_weight={dirichlet_weight:.2f})...")
    
    while any(active_mask) and completed_games < num_games:
        # Get all active boards
        active_indices = [i for i, active in enumerate(active_mask) if active]
        
        if not active_indices:
            break
            
        # Batch process all active boards for initial policy and value
        input_tensors = torch.stack([
            torch.tensor(board_to_tensor(active_games[i], move_numbers[i]), dtype=torch.float32)
            for i in active_indices
        ]).to(device)
        
        with torch.no_grad():
            policy_logits, value_preds = model(input_tensors)
        
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
                # A full MCTS would simulate to the end, but that's expensive
                sim_tensor = torch.tensor(board_to_tensor(sim_board, move_numbers[i] + 1), dtype=torch.float32).unsqueeze(0).to(device)
                
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
            board_histories[i].append(board_to_tensor(board, move_numbers[i]))
            
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
    
    if use_mcts:
        # Use MCTS for higher quality games (but slower generation)
        for i in range(num_games):
            if i % 5 == 0:
                print(f"Generating MCTS game {i+1}/{num_games}")
            game = generate_mcts_game(model, device, temperature=1.0, 
                                    num_simulations=500, c_puct=1.0, 
                                    parallel_workers=4)
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
            torch.tensor(board_to_tensor(active_games[i], move_numbers[i]), dtype=torch.float32)
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


def run_self_play_training(model, device, save_path, state_file, num_games=500, num_iterations=5, use_mcts=False):
    """Run self-play training to improve the model through reinforcement learning"""
    print(f"\n=== STARTING SELF-PLAY REINFORCEMENT LEARNING ===")
    print(f"Training for {num_iterations} iterations with {num_games} games per iteration")
    print(f"MCTS for move selection: {'Enabled' if use_mcts else 'Disabled'}")
    
    # Initialize optimizer and loss functions
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    policy_loss_fn = torch.nn.CrossEntropyLoss()
    value_loss_fn = torch.nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Get optimal batch size for this hardware
    from utils import get_optimal_batch_size
    batch_size = get_optimal_batch_size(model, device, starting_size=32, min_size=8) // 2  # Smaller for RL
    
    # Track progress across iterations
    total_positions = 0
    best_accuracy = 0
    
    # Progressive weight adjustment - start with policy focus, gradually increase value focus
    initial_policy_weight = 2.0
    initial_value_weight = 1.0
    final_value_weight = 2.5  # Increased importance of value over time
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration+1}/{num_iterations}")
        
        # Calculate current weights - increase value weight as training progresses
        progress_factor = iteration / num_iterations
        policy_weight = initial_policy_weight
        value_weight = initial_value_weight + progress_factor * (final_value_weight - initial_value_weight)
        
        print(f"Current weights: policy={policy_weight:.2f}, value={value_weight:.2f}")
        
        if use_mcts:
            # Generate games using MCTS if enabled (higher quality but slower)
            print(f"Generating {num_games} self-play games using MCTS...")
            
            # Adjust num_games if using MCTS as it's much slower
            adjusted_games = max(20, num_games // 5)
            games = generate_self_play_games(model, device, num_games=adjusted_games, use_mcts=True)
            if games:
                # Convert games to training samples
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
                        board_history.append(board_to_tensor(board, move_number))
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
            else:
                print("Failed to generate valid self-play games with MCTS. Skipping iteration.")
                continue
        else:
            # Phase 1: Generate self-play games with reward shaping for checkmate
            print(f"Generating {num_games} self-play games...")
            self_play_samples = generate_reinforcement_learning_samples(
                model,
                device, 
                num_games=num_games, 
                reward_shaping=True,
                iteration=iteration,
                total_iterations=num_iterations
            )
        
        if not self_play_samples:
            print("Failed to generate valid self-play samples. Skipping iteration.")
            continue
            
        print(f"Generated {len(self_play_samples)} training positions from self-play")
        
        # Phase 2: Train on the generated positions
        print("Training on self-play positions...")
        rl_dataset = SelfPlayDataset(self_play_samples)
        rl_dataloader = torch.utils.data.DataLoader(
            rl_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        # Training loop for this iteration
        model.train()
        total_rl_loss = 0
        batch_count = 0
        
        for batch in rl_dataloader:
            inputs, policy_targets, value_targets = batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            
            optimizer.zero_grad()
            
            if scaler:  # Use mixed precision if available
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
            
            if batch_count % 10 == 0:
                print(f"  Batch {batch_count}, Loss: {loss.item():.4f}")
        
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
        
        # Clean up between iterations
        del self_play_samples
        del rl_dataset
        del rl_dataloader
        clear_memory()
        
        total_positions += len(self_play_samples)
    
    print(f"\n=== SELF-PLAY TRAINING COMPLETED ===")
    print(f"Processed {total_positions} positions across {num_iterations} iterations")
    print(f"Best tactical accuracy: {best_accuracy:.2%}")
    return model
