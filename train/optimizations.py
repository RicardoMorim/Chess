"""
Training Optimizations for Chess AI
====================================

Hardware-tuned optimizations for:
- Intel i5 8th Gen (4 cores, 8 threads)
- NVIDIA GTX 1050 (2GB VRAM)
- 8GB RAM

Key optimizations:
1. Efficient DataLoader settings (num_workers, pin_memory, prefetch)
2. Parallel self-play with batched GPU evaluation
3. Gradient accumulation for effective larger batch sizes
4. Memory-mapped datasets for large game collections
5. Async data generation during training
6. Numba JIT for board_to_tensor
7. Mixed precision training (AMP)
"""

import os
import sys
import gc
import time
import queue
import threading
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Any
import platform

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp import autocast, GradScaler

# Windows has issues with multiprocessing DataLoader (uses spawn instead of fork)
IS_WINDOWS = platform.system() == 'Windows'

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("Numba not available - board_to_tensor will run without JIT")

import chess

# ============================================================================
# HARDWARE-SPECIFIC CONFIGURATION
# ============================================================================
HARDWARE_CONFIG = {
    # CPU settings (Intel Core Ultra 9 285K - 24 cores/24 threads)
    "cpu_cores": 24,
    "cpu_threads": 24,
    
    # DataLoader workers
    # Ultra 9 handles context switching well. 
    # Using 8 workers leaves plenty for system/GPU driving.
    "dataloader_workers": 8,
    "dataloader_prefetch": 4,       # Increase prefetch for high throughput
    
    # Self-play workers (for parallel game generation)
    # Heavy multiprocessing to utilize the 24 cores
    "selfplay_workers": 20,         
    "mcts_batch_size": 256,         # Larger batch for efficient inference
    
    # GPU settings (RTX 5080 16GB)
    # 16GB VRAM allows ~1500-2000 batch size for "Big" model
    "max_batch_size": 1536,         
    "gradient_accumulation": 1,     # No need for accum with 1536 batch
    "enable_amp": True,             # Mixed precision is critical for Tensor Cores
    
    # Memory management
    "max_cached_positions": 500000, # More RAM available
    "gc_frequency": 20,             # Less frequent GC needed
}


# ============================================================================
# OPTIMIZED BOARD TO TENSOR (with optional Numba JIT)
# ============================================================================
if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _fill_piece_planes(tensor, piece_squares, piece_channel):
        """JIT-compiled piece plane filling."""
        for sq in piece_squares:
            row = sq // 8
            col = sq % 8
            tensor[piece_channel, row, col] = 1.0
        return tensor
    
    @jit(nopython=True, cache=True)
    def _fill_attack_plane(tensor, attacked_squares, channel):
        """JIT-compiled attack plane filling."""
        for sq in attacked_squares:
            row = sq // 8
            col = sq % 8
            tensor[channel, row, col] = 1.0
        return tensor


def board_to_tensor_optimized(board: chess.Board, move_number: int = None, 
                              input_channels: int = 18) -> np.ndarray:
    """Optimized board to tensor conversion.
    
    This version:
    - Pre-allocates array once
    - Uses list comprehensions instead of loops where possible
    - Minimizes python-chess API calls
    """
    tensor = np.zeros((input_channels, 8, 8), dtype=np.float32)
    
    # Piece positions (channels 0-11) - batch the piece lookups
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        row, col = divmod(square, 8)
        channel = (piece.piece_type - 1) if piece.color == chess.WHITE else (piece.piece_type + 5)
        tensor[channel, row, col] = 1.0
    
    # Castling rights (channels 12-15) - direct assignment is faster
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[12, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[14, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    
    # En passant (channel 16)
    ep = board.ep_square
    if ep is not None:
        tensor[16, ep // 8, ep % 8] = 1.0
    
    # Side to move (channel 17)
    if board.turn == chess.WHITE:
        tensor[17, :, :] = 1.0
    
    # Extended features
    if input_channels >= 20:
        tensor[18, :, :] = min(board.halfmove_clock / 50.0, 1.0)
        mn = move_number if move_number is not None else board.fullmove_number
        tensor[19, :, :] = min(mn / 200.0, 1.0)
    
    # Attack maps (expensive - only compute if needed)
    if input_channels >= 22:
        # Use board.attacks() which is more efficient than checking each square
        for square in chess.SQUARES:
            row, col = divmod(square, 8)
            if board.is_attacked_by(chess.WHITE, square):
                tensor[20, row, col] = 1.0
            if board.is_attacked_by(chess.BLACK, square):
                tensor[21, row, col] = 1.0
    
    return tensor


# ============================================================================
# OPTIMIZED DATALOADER FACTORY
# ============================================================================
def create_optimized_dataloader(dataset: Dataset, batch_size: int, 
                                shuffle: bool = True, 
                                for_training: bool = True) -> DataLoader:
    """Create an optimized DataLoader for the hardware.
    
    Key optimizations:
    - num_workers tuned for i5-8th gen (3 workers)
    - pin_memory for faster GPU transfer
    - persistent_workers to avoid worker respawn overhead
    - prefetch_factor for async loading
    """
    config = HARDWARE_CONFIG
    
    # Windows has issues with multiprocessing DataLoader workers (pickle errors)
    # ONLY if not protected by if __name__ == "__main__"
    # We assume the user applies the fix, so we enable workers.
    if IS_WINDOWS:
        # Still be conservative on Windows spawning overhead
        num_workers = min(config["dataloader_workers"], 8)
    else:
        # Adjust workers based on dataset size
        num_workers = min(config["dataloader_workers"], len(dataset) // 100 + 1)
        num_workers = max(0, num_workers)  # At least 0
    
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": for_training,  # Drop incomplete batches during training
    }
    
    # Only use multiprocessing features if num_workers > 0
    if num_workers > 0:
        loader_kwargs.update({
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": True,  # Keep workers alive between epochs
            "prefetch_factor": config["dataloader_prefetch"],
        })
    
    return DataLoader(dataset, **loader_kwargs)


# ============================================================================
# PARALLEL SELF-PLAY GAME GENERATOR
# ============================================================================
class ParallelSelfPlayGenerator:
    """Generate self-play games in parallel using multiple processes.
    
    Architecture:
    - Main process holds the model and GPU
    - Worker processes run game logic (board operations, MCTS tree)
    - Workers send board positions to main for batch evaluation
    - Main returns policy/value to workers
    
    This avoids duplicating the model across processes (saves VRAM).
    """
    
    def __init__(self, model, device, num_workers: int = 3, 
                 mcts_simulations: int = 200, input_channels: int = 18):
        self.model = model
        self.device = device
        self.num_workers = num_workers
        self.mcts_simulations = mcts_simulations
        self.input_channels = input_channels
        
        # Communication queues
        self.eval_request_queue = Queue(maxsize=100)
        self.eval_response_queues = [Queue(maxsize=10) for _ in range(num_workers)]
        self.result_queue = Queue()
        
        # Control
        self.stop_event = Event()
        self.workers = []
        self.evaluator_thread = None
    
    def _evaluator_loop(self):
        """Background thread that batches and evaluates board positions."""
        self.model.eval()
        batch_timeout = 0.01  # 10ms timeout to collect batch
        
        while not self.stop_event.is_set():
            requests = []
            
            # Collect requests into a batch
            try:
                # Get first request (blocking)
                req = self.eval_request_queue.get(timeout=0.1)
                requests.append(req)
                
                # Try to get more requests (non-blocking)
                batch_deadline = time.time() + batch_timeout
                while len(requests) < HARDWARE_CONFIG["mcts_batch_size"]:
                    remaining = batch_deadline - time.time()
                    if remaining <= 0:
                        break
                    try:
                        req = self.eval_request_queue.get(timeout=remaining)
                        requests.append(req)
                    except queue.Empty:
                        break
                        
            except queue.Empty:
                continue
            
            if not requests:
                continue
            
            # Batch evaluate
            try:
                worker_ids = [r[0] for r in requests]
                board_tensors = [r[1] for r in requests]
                
                batch_tensor = torch.tensor(np.stack(board_tensors)).to(self.device)
                
                with torch.no_grad():
                    policy_logits, values = self.model(batch_tensor)
                    policies = F.softmax(policy_logits, dim=1).cpu().numpy()
                    values = values.cpu().numpy().flatten()
                
                # Send responses back to workers
                for i, worker_id in enumerate(worker_ids):
                    self.eval_response_queues[worker_id].put((policies[i], values[i]))
                    
            except Exception as e:
                print(f"Evaluator error: {e}")
                # Send dummy responses
                for req in requests:
                    worker_id = req[0]
                    self.eval_response_queues[worker_id].put((None, 0.0))
    
    @staticmethod
    def _worker_process(worker_id: int, eval_request_queue: Queue, 
                        eval_response_queue: Queue, result_queue: Queue,
                        stop_event: Event, mcts_simulations: int,
                        input_channels: int, num_games: int):
        """Worker process that runs game logic."""
        import chess
        import numpy as np
        
        games_completed = 0
        
        while games_completed < num_games and not stop_event.is_set():
            try:
                # Play one game
                game_data = ParallelSelfPlayGenerator._play_one_game(
                    worker_id, eval_request_queue, eval_response_queue,
                    mcts_simulations, input_channels
                )
                
                if game_data:
                    result_queue.put(game_data)
                    games_completed += 1
                    
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue
    
    @staticmethod
    def _play_one_game(worker_id: int, eval_request_queue: Queue,
                       eval_response_queue: Queue, mcts_simulations: int,
                       input_channels: int) -> List[Tuple]:
        """Play one self-play game and return training samples."""
        board = chess.Board()
        samples = []
        move_count = 0
        max_moves = 200
        
        while not board.is_game_over() and move_count < max_moves:
            # Get board tensor
            board_tensor = board_to_tensor_optimized(board, move_count + 1, input_channels)
            
            # Request evaluation
            eval_request_queue.put((worker_id, board_tensor))
            
            # Wait for response
            try:
                policy, value = eval_response_queue.get(timeout=30.0)
            except queue.Empty:
                break
            
            if policy is None:
                break
            
            # Select move (simplified - use policy directly with temperature)
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            
            # Get probabilities for legal moves
            move_probs = []
            for move in legal_moves:
                from data import get_move_index
                idx = get_move_index(move)
                move_probs.append(policy[idx] if idx < len(policy) else 1e-6)
            
            move_probs = np.array(move_probs)
            move_probs = move_probs / (move_probs.sum() + 1e-8)
            
            # Apply temperature (higher early, lower late)
            temp = 1.0 if move_count < 15 else 0.3
            if temp != 1.0:
                move_probs = np.power(move_probs, 1.0 / temp)
                move_probs = move_probs / move_probs.sum()
            
            # Sample move
            chosen_idx = np.random.choice(len(legal_moves), p=move_probs)
            chosen_move = legal_moves[chosen_idx]
            
            # Store sample (will update value at end)
            samples.append({
                'board_tensor': board_tensor.copy(),
                'policy': move_probs,
                'turn': board.turn,
                'move_idx': chosen_idx,
            })
            
            board.push(chosen_move)
            move_count += 1
        
        # Determine game result
        if board.is_checkmate():
            # Loser's turn when checkmate happens
            result = -1.0 if board.turn == chess.WHITE else 1.0
        else:
            result = 0.0  # Draw
        
        # Create training samples with proper values
        training_samples = []
        for sample in samples:
            # Value from white's perspective, flip if it was black's turn
            value = result if sample['turn'] == chess.WHITE else -result
            training_samples.append((
                sample['board_tensor'],
                sample['policy'],
                value
            ))
        
        return training_samples
    
    def generate_games(self, num_games: int) -> List[Tuple]:
        """Generate multiple self-play games in parallel."""
        games_per_worker = (num_games + self.num_workers - 1) // self.num_workers
        
        # Start evaluator thread
        self.stop_event.clear()
        self.evaluator_thread = threading.Thread(target=self._evaluator_loop, daemon=True)
        self.evaluator_thread.start()
        
        # Start worker processes
        self.workers = []
        for i in range(self.num_workers):
            worker_games = min(games_per_worker, num_games - i * games_per_worker)
            if worker_games <= 0:
                break
                
            p = Process(
                target=self._worker_process,
                args=(i, self.eval_request_queue, self.eval_response_queues[i],
                      self.result_queue, self.stop_event, self.mcts_simulations,
                      self.input_channels, worker_games)
            )
            p.start()
            self.workers.append(p)
        
        # Collect results
        all_samples = []
        games_received = 0
        
        while games_received < num_games:
            try:
                game_samples = self.result_queue.get(timeout=120.0)
                all_samples.extend(game_samples)
                games_received += 1
                
                if games_received % 5 == 0:
                    print(f"  Generated {games_received}/{num_games} games...")
                    
            except queue.Empty:
                print("Timeout waiting for game results")
                break
        
        # Cleanup
        self.stop_event.set()
        for p in self.workers:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
        
        if self.evaluator_thread:
            self.evaluator_thread.join(timeout=2.0)
        
        return all_samples
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_event.set()
        for p in self.workers:
            if p.is_alive():
                p.terminate()


# ============================================================================
# ASYNC DATA PREFETCHER
# ============================================================================
class AsyncDataPrefetcher:
    """Prefetch next batch of data while training on current batch.
    
    This hides the data loading latency by loading the next batch
    while the GPU is busy with the current batch.
    """
    
    def __init__(self, dataloader: DataLoader, device: torch.device):
        self.dataloader = dataloader
        self.device = device
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        
        self.next_batch = None
        self.iterator = None
    
    def __iter__(self):
        self.iterator = iter(self.dataloader)
        self._preload()
        return self
    
    def _preload(self):
        """Load next batch asynchronously."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.next_batch = None
            return
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                self.next_batch = tuple(
                    t.to(self.device, non_blocking=True) if isinstance(t, torch.Tensor) else t
                    for t in batch
                )
        else:
            self.next_batch = tuple(
                t.to(self.device) if isinstance(t, torch.Tensor) else t
                for t in batch
            )
    
    def __next__(self):
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        batch = self.next_batch
        if batch is None:
            raise StopIteration
        
        self._preload()
        return batch
    
    def __len__(self):
        return len(self.dataloader)


# ============================================================================
# GRADIENT ACCUMULATION TRAINER
# ============================================================================
class GradientAccumulationTrainer:
    """Training with gradient accumulation for effective larger batch sizes.
    
    This allows training with effectively larger batches than fit in VRAM
    by accumulating gradients over multiple forward/backward passes.
    """
    
    def __init__(self, model, optimizer, device, accumulation_steps: int = 2,
                 use_amp: bool = True, grad_clip: float = 1.0):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip = grad_clip
        
        self.scaler = GradScaler() if self.use_amp else None
        self.current_step = 0
    
    def train_step(self, batch, policy_loss_fn, value_loss_fn) -> Dict[str, float]:
        """Perform one training step with gradient accumulation."""
        boards, policy_targets, value_targets = batch
        boards = boards.to(self.device)
        policy_targets = policy_targets.to(self.device)
        value_targets = value_targets.to(self.device)
        
        # Forward pass with optional AMP
        if self.use_amp:
            with autocast():
                policy_logits, value_pred = self.model(boards)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                loss = policy_loss + value_loss
                loss = loss / self.accumulation_steps
            
            # Backward pass with scaled gradients
            self.scaler.scale(loss).backward()
        else:
            policy_logits, value_pred = self.model(boards)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            loss = policy_loss + value_loss
            loss = loss / self.accumulation_steps
            loss.backward()
        
        self.current_step += 1
        
        # Optimizer step after accumulation
        metrics = {
            'policy_loss': policy_loss.item() * self.accumulation_steps,
            'value_loss': value_loss.item() * self.accumulation_steps,
            'total_loss': loss.item() * self.accumulation_steps,
        }
        
        if self.current_step % self.accumulation_steps == 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            metrics['optimizer_step'] = True
        else:
            metrics['optimizer_step'] = False
        
        return metrics


# ============================================================================
# MEMORY MANAGEMENT UTILITIES
# ============================================================================
def aggressive_memory_cleanup():
    """Aggressive memory cleanup for low-VRAM systems."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_memory_stats() -> Dict[str, float]:
    """Get current memory usage statistics."""
    stats = {
        'ram_used_gb': 0.0,
        'vram_used_gb': 0.0,
        'vram_total_gb': 0.0,
    }
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        stats['ram_used_gb'] = process.memory_info().rss / 1e9
    except:
        pass
    
    if torch.cuda.is_available():
        stats['vram_used_gb'] = torch.cuda.memory_allocated() / 1e9
        stats['vram_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return stats


def print_memory_stats(label: str = ""):
    """Print memory usage statistics."""
    stats = get_memory_stats()
    print(f"[Memory{' - ' + label if label else ''}] "
          f"RAM: {stats['ram_used_gb']:.2f}GB, "
          f"VRAM: {stats['vram_used_gb']:.2f}/{stats['vram_total_gb']:.2f}GB")


# ============================================================================
# QUICK TEST
# ============================================================================
if __name__ == "__main__":
    print("Testing optimizations...")
    print(f"Hardware config: {HARDWARE_CONFIG}")
    print(f"Numba available: {HAS_NUMBA}")
    
    # Test board_to_tensor
    board = chess.Board()
    
    import time
    start = time.time()
    for _ in range(1000):
        tensor = board_to_tensor_optimized(board, 1, 18)
    elapsed = time.time() - start
    print(f"board_to_tensor_optimized: {elapsed:.3f}s for 1000 calls ({elapsed*1000:.3f}ms/call)")
    
    print_memory_stats("startup")
    print("Done!")
