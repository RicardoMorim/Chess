"""
GPU-Accelerated MCTS Self-Play Adapter
======================================

Enables CPU workers to evaluate board positions via a GPU inference server.

Instead of each worker holding its own model copy (expensive GPU memory),
all workers send board evaluation requests to a single GPU server that
batches them for efficient forward passes.

Usage:
    # In league trainer:
    server = create_inference_server(model)  # Start GPU server
    
    # Pass server to workers:
    p = Process(target=worker_with_gpu_eval, args=(..., server.evaluate_async, ...))
    
    # Workers call:
    policy, value = server.evaluate(board)  # Or evaluate_async() for pipelined eval

Note: This is an OPTIONAL optimization layer. The league trainer can run
without GPU-batched eval (existing CPU-only MCTS still works). This adds
throughput under GPU constraint (e.g., when model is large or batch size is small).
"""

import logging
import chess
from typing import Tuple, Optional, Callable
import torch

logger = logging.getLogger(__name__)


def create_gpu_eval_wrapper(
    evaluate_fn: Callable[[chess.Board], Tuple[torch.Tensor, float]],
) -> Callable[[chess.Board], Tuple[torch.Tensor, float]]:
    """
    Wrap a synchronous evaluate function (local or remote) for MCTS use.
    
    Args:
        evaluate_fn: Function that returns (policy_logits, value) for a board
    
    Returns:
        Wrapped evaluate function safe for use in MCTS
    """
    def wrapped_evaluate(board: chess.Board) -> Tuple[torch.Tensor, float]:
        try:
            policy, value = evaluate_fn(board)
            # Clamp value to valid range
            value = float(torch.clamp(torch.tensor(value), -1.0, 1.0))
            return policy, value
        except Exception as e:
            logger.error(f"GPU eval failed: {e}. Returning neutral estimate.")
            legal_moves = list(board.legal_moves)
            return (
                torch.log_softmax(torch.zeros(len(legal_moves)), dim=0),
                0.0  # Neutral value
            )
    
    return wrapped_evaluate


class LocalGPUEvaluator:
    """
    Local GPU evaluator: loads model on current thread and evaluates boards.
    
    Used to integrate GPU evaluation into worker processes (if GPU is available
    and model is small enough to replicate).
    """
    
    def __init__(self, model, device: str = "cuda"):
        """
        Initialize local GPU evaluator.
        
        Args:
            model: PyTorch model
            device: Device for inference
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def __call__(self, board: chess.Board) -> Tuple[torch.Tensor, float]:
        """
        Evaluate a board.
        
        Args:
            board: Chess position
        
        Returns:
            (policy_logits_for_legal_moves, value)
        """
        try:
            # TODO: Implement board → features conversion
            # For now, return dummy results
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                return torch.zeros(1), 0.0
            
            with torch.no_grad():
                # Dummy forward pass
                policy = torch.log_softmax(torch.zeros(len(legal_moves)), dim=0)
                value = 0.0
            
            return policy, value
        except Exception as e:
            logger.error(f"Local GPU eval failed: {e}")
            legal_moves = list(board.legal_moves)
            return torch.log_softmax(torch.zeros(len(legal_moves)), dim=0), 0.0


# Integration points for MCTS:
# If using GPU-batched server (advanced):
#   evaluate_fn = server.evaluate  or  server.evaluate_async
# If using local GPU (simple):
#   evaluate_fn = LocalGPUEvaluator(model)
# If using CPU only (current):
#   evaluate_fn = mcts.generate_mcts_game with default CPU MCTS
