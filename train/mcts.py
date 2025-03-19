import time
import numpy as np
import torch
import torch.nn.functional as F
import chess
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional, List, Any

from data import board_to_tensor, get_move_index

from utils import clear_memory

# Define a node class for MCTS.
class MCTSNode:
    def __init__(self, board: chess.Board, prior: float, parent=None):
        self.board = board.copy()
        self.prior = prior         # Prior probability from the neural network.
        self.parent = parent
        self.children = {}         # Dictionary: move -> MCTSNode.
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0      # For parallelization.
        self.lock = threading.Lock()  # Ensure thread-safe updates.

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0

# Expand a leaf node using the neural network.
def expand_node(node: MCTSNode, model, device):
    board_tensor = torch.tensor(board_to_tensor(node.board, node.board.fullmove_number), 
                                  dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)
    policy = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()
    
    legal_moves = list(node.board.legal_moves)
    for move in legal_moves:
        next_board = node.board.copy()
        next_board.push(move)
        move_index = get_move_index(move)
        node.children[move] = MCTSNode(next_board, prior=policy[move_index], parent=node)
    return float(value_pred.item())

# Select a child based on the PUCT (Polynomial Upper Confidence Trees) formula.
def select_child(node: MCTSNode, c_puct: float):
    best_score = -float("inf")
    best_move = None
    best_child = None
    with node.lock:
        total_visits = np.sqrt(node.visit_count + 1)
        for move, child in node.children.items():
            with child.lock:
                # Incorporate virtual loss in the denominator.
                q_value = child.value() if child.visit_count > 0 else 0
                ucb = q_value + c_puct * child.prior * total_visits / (1 + child.visit_count + child.virtual_loss)
            if ucb > best_score:
                best_score = ucb
                best_move = move
                best_child = child
    return best_move, best_child

# Recursively simulate a game from the current node.
def simulate(node: MCTSNode, model, device, c_puct: float, virtual_loss: float = 1.0):
    # Terminal condition: if the game is over, return the result.
    if node.board.is_game_over():
        if node.board.is_checkmate():
            # Return +1 if the side not to move has delivered checkmate.
            return 1.0 if not node.board.turn else -1.0
        else:
            return 0.0

    # Expand the node if it is a leaf.
    with node.lock:
        if not node.children:
            value = expand_node(node, model, device)
            node.visit_count += 1
            node.value_sum += value
            return value

    # Select the best child according to UCB.
    move, child = select_child(node, c_puct)
    
    # Apply virtual loss to discourage duplicate exploration in parallel.
    with child.lock:
        child.virtual_loss += virtual_loss

    # Recursively simulate from the child.
    value = -simulate(child, model, device, c_puct, virtual_loss)

    # Remove the virtual loss and update statistics.
    with node.lock:
        node.visit_count += 1
        node.value_sum += value
    with child.lock:
        child.virtual_loss -= virtual_loss

    return value

# Run MCTS with dynamic simulation budget and parallelization.
def run_mcts(root_board: chess.Board, model, device, num_simulations: int = None, time_limit: float = None, 
             c_puct: float = 1.0, virtual_loss: float = 1.0, parallel_workers: int = 4):
    root = MCTSNode(root_board, prior=1.0)
    expand_node(root, model, device)

    early_stopping_threshold = 0.9  # Stop if one move has 90% of visits
    minimum_visits = 100  # But only after this many visits
    visits_without_improvement = 0  # Track non-improving iterations
    best_move_visits = 0
    
    simulations_done = 0
    start_time = time.time()
    
    def run_one_simulation():
        nonlocal simulations_done
        simulate(root, model, device, c_puct, virtual_loss)
        simulations_done += 1

    # Determine the total simulation budget (either fixed number or dynamic by time).
    total_simulations = num_simulations if num_simulations is not None else float('inf')
    
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            while simulations_done < total_simulations:
                # Add batched simulation - this is faster than one-by-one
                batch_size = parallel_workers * 2
                futures = [executor.submit(run_one_simulation) for _ in range(min(batch_size, total_simulations - simulations_done))]
                
                for future in futures:
                    future.result()
                    
                # Early stopping check (periodically)
                if simulations_done >= minimum_visits and simulations_done % 20 == 0:
                    visits = {move: child.visit_count for move, child in root.children.items()}
                    if visits:
                        top_move = max(visits.items(), key=lambda x: x[1])[0]
                        top_visits = visits[top_move]
                        total = sum(visits.values())
                        
                        if top_visits > early_stopping_threshold * total:
                            # One move is dominating, stop early
                            break
                            
                        # Detect when search is not improving
                        if top_visits <= best_move_visits:
                            visits_without_improvement += 1
                            if visits_without_improvement >= 10:  # Multiple non-improving iterations
                                break
                        else:
                            best_move_visits = top_visits
                            visits_without_improvement = 0
        
    # Gather visit counts for all legal moves at the root.
    visit_counts = {}
    for move, child in root.children.items():
        with child.lock:
            visit_counts[move] = child.visit_count
    return visit_counts, root

# Update the MCTS tree for persistence across moves.
def update_tree(root: MCTSNode, chosen_move):
    if chosen_move in root.children:
        new_root = root.children[chosen_move]
        new_root.parent = None  # Detach from previous tree.
        return new_root
    return None

# Use the improved MCTS to select a move.
def select_move_with_mcts(board: chess.Board, model, device, num_simulations: int = 100, 
                            time_limit: float = None, temperature: float = 1.0, c_puct: float = 1.5, 
                            virtual_loss: float = 1.0, parallel_workers: int = 4, tree: MCTSNode = None):
    # If a persistent tree exists and matches the current board, reuse it.
    if tree is not None and tree.board.fen() == board.fen():
        root = tree
    else:
        root = MCTSNode(board, prior=1.0)
        expand_node(root, model, device)
    
    visit_counts, root = run_mcts(board, model, device, num_simulations=num_simulations, 
                                  time_limit=time_limit, c_puct=c_puct, virtual_loss=virtual_loss, 
                                  parallel_workers=parallel_workers)
    
    # Convert visit counts into a probability distribution.
    moves = list(visit_counts.keys())
    if not moves:  # No legal moves
        return None, np.array([]), None
        
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float32)
    if temperature == 0:
        best_idx = np.argmax(counts)
        best_move = moves[best_idx]
        pi = np.zeros_like(counts)
        pi[best_idx] = 1.0
    else:
        counts = np.power(counts, 1.0 / temperature)
        pi = counts / np.sum(counts)
        best_move = np.random.choice(moves, p=pi)
    
    # For tree persistence, update the tree to the chosen move.
    new_tree = update_tree(root, best_move)
    
    return best_move, pi, new_tree

# Function to enhance self-play by using MCTS instead of simpler search
def generate_mcts_game(model, device, temperature=1.0, num_simulations=50, 
                      c_puct=1.5, parallel_workers=2):
    """Generate a self-play game using MCTS for move selection with more stable settings"""
    game = chess.pgn.Game()
    board = chess.Board()
    node = game
    move_number = 1
    tree = None

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
            
        # Select move using MCTS
        try:
            move, _, tree = select_move_with_mcts(
                board, 
                model, 
                device,
                num_simulations=num_simulations,
                temperature=current_temp,
                c_puct=c_puct,
                parallel_workers=parallel_workers, 
                tree=tree,
                time_limit=5.0  # Add time limit to avoid hangs
            )
        except Exception as e:
            print(f"MCTS error: {e}. Falling back to direct move selection.")
            # Fallback to direct move selection if MCTS fails
            input_tensor = torch.tensor(board_to_tensor(board, move_number), dtype=torch.float32).unsqueeze(0).to(device)
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
