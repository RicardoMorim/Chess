# ============================================================================
# MCTS NODE
# ============================================================================
import time, threading, numpy as np, torch, chess
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Dict, Tuple, Optional
from .data import board_to_tensor, get_move_index
from .constants import ACTION_SPACE_SIZE, MCTS_CONFIG
from .utils import clear_memory

# ------------------------
# Node
# ------------------------
class MCTSNode:
    __slots__ = ['fen', 'prior', 'parent', 'children', 'visit_count', 'value_sum', 'virtual_loss', 'lock']
    def __init__(self, board: chess.Board, prior: float, parent=None):
        self.fen = board.fen()
        self.prior = prior
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.virtual_loss = 0
        self.lock = threading.Lock()

    def get_board(self):
        return chess.Board(self.fen)

    def value(self):
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    def is_expanded(self):
        return len(self.children) > 0

# ------------------------
# Batched Node Expansion
# ------------------------
def expand_nodes_batched(nodes: List[MCTSNode], model, device, add_noise=True):
    if not nodes: return []

    input_channels = getattr(model, "input_channels", 22)
    boards = np.stack([board_to_tensor(n.get_board(), n.get_board().fullmove_number, input_channels) for n in nodes])
    boards_tensor = torch.tensor(boards, dtype=torch.float32).to(device)

    with torch.no_grad():
        policy_logits, values = model(boards_tensor)
        policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
        values = values.cpu().numpy().flatten()

    for idx, node in enumerate(nodes):
        legal_moves = list(node.get_board().legal_moves)
        if not legal_moves:
            continue
        priors = np.array([policy_probs[idx][get_move_index(m)] for m in legal_moves], dtype=np.float32)
        priors /= (priors.sum() + 1e-8)
        # Dirichlet noise at root
        if add_noise:
            noise = np.random.dirichlet([0.3]*len(legal_moves))
            priors = 0.75*priors + 0.25*noise
        for move, prior in zip(legal_moves, priors):
            child_board = node.get_board().copy()
            child_board.push(move)
            node.children[move] = MCTSNode(child_board, prior, parent=node)
    return values

# ------------------------
# Batched Simulation
# ------------------------
def simulate_batched(root: MCTSNode, model, device, c_puct=2.5, virtual_loss=1.0):
    """Run one simulation from root, batched across leaves."""
    path = []
    node = root
    # Selection
    while node.is_expanded():
        best_score = -float("inf")
        best_child = None
        sqrt_parent_visits = np.sqrt(node.visit_count + 1)
        for move, child in list(node.children.items()):
            with child.lock:
                q_val = child.value()
                vis = child.visit_count
                vl = child.virtual_loss
                prior = child.prior
            u = c_puct * prior * sqrt_parent_visits / (1 + vis + vl)
            score = q_val + u
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None: break
        path.append(best_child)
        with best_child.lock:
            best_child.virtual_loss += virtual_loss
        node = best_child

    # Expansion + Evaluation
    if not node.is_expanded() and not node.get_board().is_game_over():
        value = expand_nodes_batched([node], model, device, add_noise=True)[0]
    else:
        value = 0.0 if node.get_board().is_game_over() else node.value()

    # Backpropagate
    for n in [root]+path[::-1]:
        with n.lock:
            n.value_sum += value
            n.visit_count += 1
            if n != root:
                n.virtual_loss -= virtual_loss
        value = -value  # alternate sides

# ------------------------
# Batched MCTS Runner
# ------------------------
def run_mcts_batched(root_board: chess.Board, model, device, num_simulations=800, parallel_workers=24):
    root = MCTSNode(root_board, prior=1.0)
    expand_nodes_batched([root], model, device, add_noise=True)

    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        futures = [executor.submit(simulate_batched, root, model, device) for _ in range(num_simulations)]
        for f in futures:
            f.result()
    
    # Collect move counts
    visit_counts = {m: c.visit_count for m, c in root.children.items()}
    return visit_counts, root

# ------------------------
# Batched Move Selection
# ------------------------
def select_move(root_board: chess.Board, model, device, num_simulations=800, temperature=1.0, parallel_workers=24):
    visit_counts, root = run_mcts_batched(root_board, model, device, num_simulations, parallel_workers)
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float32)
    if temperature == 0:
        best_idx = np.argmax(counts)
        return moves[best_idx], root
    counts_temp = np.power(counts + 1e-8, 1.0/temperature)
    probs = counts_temp / counts_temp.sum()
    return np.random.choice(moves, p=probs), root


# ============================================================================
# NODE EXPANSION
# ============================================================================
def expand_node(
    node: MCTSNode,
    model,
    device,
    add_noise: bool = False,
    evaluate_fn: Optional[Callable[[chess.Board], Tuple[Optional[List[float]], float]]] = None,
) -> float:
    """Expand a leaf node using the neural network.
    
    Args:
        node: The node to expand
        model: Neural network model
        device: Computation device
        add_noise: Whether to add Dirichlet noise (for root node)
    
    Returns:
        The value estimate for this position
    """
    legal_moves = list(node.get_board().legal_moves)

    # External evaluation path (GPU-batched server)
    if evaluate_fn is not None:
        try:
            legal_logits, value_pred = evaluate_fn(node.get_board())
        except Exception as eval_err:
            legal_logits, value_pred = None, 0.0

        if not legal_moves:
            return float(value_pred)

        if legal_logits is None or len(legal_logits) != len(legal_moves):
            move_priors = np.ones(len(legal_moves), dtype=np.float32) / max(1, len(legal_moves))
        else:
            logits = np.asarray(legal_logits, dtype=np.float32)
            logits = logits - np.max(logits)  # stabilize
            exp = np.exp(logits)
            move_priors = exp / (np.sum(exp) + 1e-8)

        # Add Dirichlet noise at root for exploration
        if add_noise and len(legal_moves) > 0:
            noise = np.random.dirichlet([MCTS_CONFIG['dirichlet_alpha']] * len(legal_moves))
            epsilon = MCTS_CONFIG['dirichlet_epsilon']
            move_priors = (1 - epsilon) * move_priors + epsilon * noise

        # Create child nodes
        for move, prior in zip(legal_moves, move_priors):
            next_board = node.get_board().copy()
            next_board.push(move)
            node.children[move] = MCTSNode(next_board, prior=prior, parent=node)

        return float(value_pred)

    # Local model evaluation path (default)
    input_channels = model.input_channels if hasattr(model, 'input_channels') else 22

    # Generate board tensor (only 18/20/22 channels supported)
    board_tensor = torch.tensor(
        board_to_tensor(node.get_board(), node.get_board().fullmove_number, input_channels),
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        policy_logits, value_pred = model(board_tensor)

    policy = F.softmax(policy_logits, dim=1).cpu().numpy().flatten()

    if not legal_moves:
        return float(value_pred.item())

    # Extract priors for legal moves only
    move_priors = []
    for move in legal_moves:
        move_index = get_move_index(move)
        if move_index < len(policy):
            move_priors.append(policy[move_index])
        else:
            move_priors.append(1e-6)  # Small prior for unmapped moves
    move_priors = np.array(move_priors, dtype=np.float32)
    
    # Normalize priors
    prior_sum = np.sum(move_priors)
    if prior_sum > 0:
        move_priors /= prior_sum
    else:
        move_priors = np.ones(len(legal_moves)) / len(legal_moves)
    
    # Add Dirichlet noise at root for exploration
    if add_noise and len(legal_moves) > 0:
        noise = np.random.dirichlet([MCTS_CONFIG['dirichlet_alpha']] * len(legal_moves))
        epsilon = MCTS_CONFIG['dirichlet_epsilon']
        move_priors = (1 - epsilon) * move_priors + epsilon * noise
    
    # Create child nodes
    for move, prior in zip(legal_moves, move_priors):
        next_board = node.get_board().copy()
        next_board.push(move)
        node.children[move] = MCTSNode(next_board, prior=prior, parent=node)
    
    return float(value_pred.item())


# ============================================================================
# PUCT SELECTION (Corrected AlphaZero formula)
# ============================================================================
def select_child(node: MCTSNode, c_puct: float) -> Tuple[chess.Move, 'MCTSNode']:
    """Select child node using PUCT formula.
    
    AlphaZero PUCT formula:
    UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    
    Where:
    - Q(s,a) = mean value of action a from state s
    - P(s,a) = prior probability of action a
    - N(s) = visit count of parent state
    - N(s,a) = visit count of action a
    """
    best_score = -float("inf")
    best_move = None
    best_child = None
    
    with node.lock:
        sqrt_parent_visits = np.sqrt(max(1, node.visit_count))
        children_snapshot = list(node.children.items())

    for move, child in children_snapshot:
        with child.lock:
            q_val = (child.value_sum / (child.visit_count + child.virtual_loss)
                     if child.visit_count > 0 else 0.0)
            vis = child.visit_count
            vl = child.virtual_loss
            prior = child.prior
        u = c_puct * prior * sqrt_parent_visits / (1 + vis + vl)
        score = q_val + u
        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    return best_move, best_child


# ============================================================================
# MCTS SIMULATION
# ============================================================================
def simulate(
    node: MCTSNode,
    model,
    device,
    c_puct: float,
    virtual_loss: float = 1.0,
    evaluate_fn: Optional[Callable[[chess.Board], Tuple[Optional[List[float]], float]]] = None,
) -> float:
    """Run one simulation from the given node.
    
    1. Selection: Select child nodes until reaching a leaf
    2. Expansion: Expand the leaf node
    3. Backpropagation: Update values along the path
    
    Returns:
        Value from the perspective of the node's player
    """
    # Terminal check
    if node.get_board().is_game_over():
        if node.get_board().is_checkmate():
            # Checkmate: -1 for side to move (they lost)
            return -1.0
        else:
            # Draw
            return 0.0

    # Expand if leaf node
    with node.lock:
        if not node.is_expanded():
            value = expand_node(node, model, device, add_noise=False, evaluate_fn=evaluate_fn)
            node.visit_count += 1
            node.value_sum += value
            return value

    # Select best child
    move, child = select_child(node, c_puct)
    
    if child is None:
        return 0.0
    
    # Apply virtual loss
    with child.lock:
        child.virtual_loss += virtual_loss

    # Recursive simulation (negate value for opponent)
    value = -simulate(child, model, device, c_puct, virtual_loss, evaluate_fn=evaluate_fn)

    # Backpropagate and remove virtual loss
    with node.lock:
        node.visit_count += 1
        node.value_sum += value
    with child.lock:
        child.virtual_loss -= virtual_loss

    return value


# ============================================================================
# RUN MCTS
# ============================================================================
def run_mcts(
    root_board: chess.Board,
    model,
    device,
    num_simulations: int = 800,
    time_limit: float = None,
    c_puct: float = None,
    virtual_loss: float = None,
    parallel_workers: int = 4,
    add_noise: bool = True,
    evaluate_fn: Optional[Callable[[chess.Board], Tuple[Optional[List[float]], float]]] = None,
) -> Tuple[Dict, MCTSNode]:
    """Run Monte Carlo Tree Search from a root position.
    
    Args:
        root_board: Starting chess position
        model: Neural network for evaluation
        device: Computation device
        num_simulations: Number of simulations to run
        time_limit: Optional time limit in seconds
        c_puct: Exploration constant (defaults to config value)
        virtual_loss: Virtual loss for parallelization
        parallel_workers: Number of parallel workers
        add_noise: Whether to add Dirichlet noise at root
    
    Returns:
        Dictionary of move -> visit_count, and the root node
    """
    # Use config defaults
    if c_puct is None:
        c_puct = MCTS_CONFIG['c_puct']
    if virtual_loss is None:
        virtual_loss = MCTS_CONFIG['virtual_loss']
    
    # Create and expand root
    root = MCTSNode(root_board, prior=1.0)
    expand_node(root, model, device, add_noise=add_noise, evaluate_fn=evaluate_fn)
    
    if not root.children:
        return {}, root
    
    # Early stopping config
    early_stop_threshold = MCTS_CONFIG['early_stop_threshold']
    min_simulations = MCTS_CONFIG['min_simulations']
    
    simulations_done = 0
    start_time = time.time()
    
    def run_one_simulation():
        nonlocal simulations_done
        simulate(root, model, device, c_puct, virtual_loss, evaluate_fn=evaluate_fn)
        simulations_done += 1

    # Determine budget
    total_simulations = num_simulations if num_simulations is not None else float('inf')
    
    # Run simulations
    with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        while simulations_done < total_simulations:
            # Check time limit
            if time_limit is not None and (time.time() - start_time) > time_limit:
                break
            
            # Submit batch of simulations
            batch_size = min(parallel_workers * 2, int(total_simulations - simulations_done))
            if batch_size <= 0:
                break
                
            futures = [executor.submit(run_one_simulation) for _ in range(batch_size)]
            for future in futures:
                try:
                    future.result(timeout=10.0)
                except Exception as e:
                    pass  # Ignore individual simulation failures
            
            # Early stopping check
            if simulations_done >= min_simulations:
                visits = {move: child.visit_count for move, child in root.children.items()}
                if visits:
                    total_visits = sum(visits.values())
                    max_visits = max(visits.values())
                    if max_visits > early_stop_threshold * total_visits:
                        break
    
    # Gather visit counts
    visit_counts = {}
    for move, child in root.children.items():
        with child.lock:
            visit_counts[move] = child.visit_count
    
    return visit_counts, root


# ============================================================================
# TREE PERSISTENCE
# ============================================================================
def update_tree(root: MCTSNode, chosen_move: chess.Move) -> Optional[MCTSNode]:
    """Update tree after a move is made, reusing subtree if possible."""
    if chosen_move in root.children:
        new_root = root.children[chosen_move]
        new_root.parent = None
        return new_root
    return None


# ============================================================================
# MOVE SELECTION
# ============================================================================
def select_move_with_mcts(board: chess.Board, model, device, num_simulations: int = 400, 
                          time_limit: float = None, temperature: float = 1.0, c_puct: float = None, 
                          virtual_loss: float = None, parallel_workers: int = 4, 
                          tree: MCTSNode = None, add_noise: bool = True):
    """Select a move using MCTS.
    
    Args:
        board: Current chess position
        model: Neural network
        device: Computation device
        num_simulations: Number of MCTS simulations
        time_limit: Optional time limit
        temperature: Temperature for move selection (0 = greedy, 1 = proportional)
        c_puct: Exploration constant
        virtual_loss: Virtual loss for parallelization
        parallel_workers: Number of parallel workers
        tree: Optional existing tree to reuse
        add_noise: Whether to add Dirichlet noise
    
    Returns:
        (selected_move, policy_distribution, updated_tree)
    """
    # Reuse tree if possible
    if tree is not None and tree.fen == board.fen():
        root = tree
        # Re-expand if needed
        if not root.is_expanded():
            expand_node(root, model, device, add_noise=add_noise)
        visit_counts = {}
        for move, child in root.children.items():
            visit_counts[move] = child.visit_count
    else:
        # Run new MCTS
        visit_counts, root = run_mcts(
            board, model, device, 
            num_simulations=num_simulations,
            time_limit=time_limit, 
            c_puct=c_puct, 
            virtual_loss=virtual_loss, 
            parallel_workers=parallel_workers,
            add_noise=add_noise
        )
    
    moves = list(visit_counts.keys())
    if not moves:
        return None, np.array([]), None
    
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float32)
    
    # Apply temperature
    if temperature == 0 or np.sum(counts) == 0:
        # Greedy selection
        best_idx = np.argmax(counts) if np.sum(counts) > 0 else 0
        best_move = moves[best_idx]
        pi = np.zeros(len(counts))
        pi[best_idx] = 1.0
    else:
        # Temperature-scaled selection
        counts_temp = np.power(counts + 1e-8, 1.0 / temperature)
        pi = counts_temp / np.sum(counts_temp)
        best_move = np.random.choice(moves, p=pi)
    
    # Update tree
    new_tree = update_tree(root, best_move)
    
    return best_move, pi, new_tree


# ============================================================================
# GAME GENERATION
# ============================================================================
def generate_mcts_game(model, device, temperature=1.0, num_simulations=400, 
                       c_puct=None, parallel_workers=4, input_channels=None):
    """Generate a self-play game using MCTS.
    
    Uses temperature annealing:
    - Moves 1-15: temperature=1.0 (exploration)
    - Moves 16-30: temperature=0.5 (balanced)
    - Moves 31+: temperature=0.1 (exploitation)
    """
    game = chess.pgn.Game()
    board = chess.Board()
    node = game
    move_number = 1
    tree = None
    max_moves = 200
    
    if input_channels is None:
        input_channels = model.input_channels if hasattr(model, 'input_channels') else 20
    
    while not board.is_game_over() and move_number <= max_moves:
        # Temperature annealing for exploration vs exploitation
        if move_number <= 15:
            current_temp = temperature
        elif move_number <= 30:
            current_temp = temperature * 0.5
        else:
            current_temp = 0.1  # Near-greedy
        
        # Add noise only in early game
        add_noise = move_number <= 30
        
        try:
            move, _, tree = select_move_with_mcts(
                board, model, device,
                num_simulations=num_simulations,
                temperature=current_temp,
                c_puct=c_puct,
                parallel_workers=parallel_workers,
                tree=tree,
                time_limit=30.0,
                add_noise=add_noise
            )
        except Exception as e:
            print(f"MCTS error at move {move_number}: {e}")
            # Fallback to direct policy
            move = _fallback_move_selection(board, model, device, input_channels)
        
        if move is None:
            break
        
        board.push(move)
        node = node.add_variation(move)
        move_number += 1
        
        # Periodic cleanup
        if move_number % 30 == 0:
            clear_memory()
    
    # Set result
    if board.is_checkmate():
        game.headers["Result"] = "0-1" if board.turn == chess.WHITE else "1-0"
    elif board.is_stalemate() or board.is_insufficient_material():
        game.headers["Result"] = "1/2-1/2"
    elif board.is_fifty_moves() or board.is_repetition(3):
        game.headers["Result"] = "1/2-1/2"
    else:
        game.headers["Result"] = "*"
    
    return game


def _fallback_move_selection(board, model, device, input_channels):
    """Fallback move selection using direct policy network."""
    input_tensor = torch.tensor(
        board_to_tensor(board, board.fullmove_number, input_channels),
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    with torch.no_grad():
        policy_logits, _ = model(input_tensor)
    
    policy = F.softmax(policy_logits, dim=1).squeeze().cpu().numpy()
    legal_moves = list(board.legal_moves)
    
    if not legal_moves:
        return None
    
    move_probs = []
    for move in legal_moves:
        idx = get_move_index(move)
        if idx < len(policy):
            move_probs.append(policy[idx])
        else:
            move_probs.append(1e-6)
    
    move_probs = np.array(move_probs)
    if np.sum(move_probs) <= 1e-10:
        move_probs = np.ones(len(legal_moves)) / len(legal_moves)
    else:
        move_probs /= np.sum(move_probs)
    
    return np.random.choice(legal_moves, p=move_probs)


# ============================================================================
# MCTS WRAPPER CLASS (for league self-play workers)
# ============================================================================
class MCTS:
    """
    Lightweight MCTS wrapper with a simple search() API.
    Returns a full policy vector (size ACTION_SPACE_SIZE) and selected move.
    """

    def __init__(
        self,
        model,
        device,
        num_visits: int = 800,
        c_puct: float = 2.5,
        temperature: float = 1.0,
        dirichlet_alpha: float = None,
        add_noise: bool = True,
        parallel_workers: int = 4,
        virtual_loss: float = 1.0,
        evaluate_fn: Optional[Callable[[chess.Board], Tuple[Optional[List[float]], float]]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.num_visits = num_visits
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.add_noise = add_noise
        self.parallel_workers = parallel_workers
        self.virtual_loss = virtual_loss
        self.evaluate_fn = evaluate_fn

    def search(self, board: chess.Board, temperature: Optional[float] = None):
        """Run MCTS and return (policy_vector, selected_move).

        Args:
            board: Position to search from.
            temperature: Optional per-call override for the move-selection
                temperature. If None, uses the temperature passed to the
                constructor. Use 0 for greedy selection (AlphaZero: τ=0
                after the first 30 half-moves).
        """
        effective_temperature = self.temperature if temperature is None else temperature
        visit_counts, _root = run_mcts(
            board,
            self.model,
            self.device,
            num_simulations=self.num_visits,
            c_puct=self.c_puct,
            virtual_loss=self.virtual_loss,
            parallel_workers=self.parallel_workers,
            add_noise=self.add_noise,
            evaluate_fn=self.evaluate_fn,
        )

        # Store root value for resignation logic
        self._last_value = _root.value() if _root else 0.0

        legal_moves = list(visit_counts.keys())
        if not legal_moves:
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), None

        counts = np.array([visit_counts[m] for m in legal_moves], dtype=np.float32)

        # Temperature-scaled selection
        if effective_temperature == 0 or np.sum(counts) == 0:
            probs = np.zeros_like(counts)
            probs[np.argmax(counts)] = 1.0
        else:
            counts_temp = np.power(counts + 1e-8, 1.0 / effective_temperature)
            probs = counts_temp / np.sum(counts_temp)

        # Build full policy vector
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for move, p in zip(legal_moves, probs):
            idx = get_move_index(move)
            if idx < ACTION_SPACE_SIZE:
                policy[idx] = p

        selected_move = np.random.choice(legal_moves, p=probs)
        return policy, selected_move
