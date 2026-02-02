"""
Puzzle Training - Bootstrap Phase
==================================

Supervised learning on tactical puzzles to reduce cold-start time.

This runs ONCE before league training begins. It teaches the model
basic tactical patterns so self-play starts from a non-random policy.

THE RULE: After bootstrap completes, only MCTS self-play improves models.
          This code should never be called from the main training loop.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import logging
import time
from typing import Optional, Dict, Any, List, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_model, board_to_tensor, get_move_index, PolicyLoss, ValueLoss

logger = logging.getLogger(__name__)


class PuzzleDataset(Dataset):
    """
    Dataset for chess tactical puzzles.
    
    Expected format:
    - Each puzzle: (FEN, move_sequence, rating, themes)
    - First move in sequence is the puzzle solution
    """
    
    def __init__(
        self,
        puzzle_file: str,
        input_channels: int = 22,
        max_puzzles: Optional[int] = None,
    ):
        """
        Initialize puzzle dataset.
        
        Args:
            puzzle_file: Path to puzzle CSV/JSON file
            input_channels: Number of input channels for board representation
            max_puzzles: Maximum number of puzzles to load (for testing)
        """
        self.input_channels = input_channels
        self.puzzles = []
        
        self._load_puzzles(puzzle_file, max_puzzles)
        logger.info(f"Loaded {len(self.puzzles)} puzzles")
    
    def _load_puzzles(self, puzzle_file: str, max_puzzles: Optional[int]) -> None:
        """Load puzzles from file."""
        import chess
        import csv
        
        path = Path(puzzle_file)
        
        if not path.exists():
            logger.warning(f"Puzzle file not found: {puzzle_file}")
            return
        
        if path.suffix == '.csv':
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    if max_puzzles and i >= max_puzzles:
                        break
                    
                    try:
                        fen = row.get('FEN', row.get('fen', ''))
                        moves = row.get('Moves', row.get('moves', '')).split()
                        rating = int(row.get('Rating', row.get('rating', 1500)))
                        
                        if fen and moves:
                            # Create board and apply opponent's move
                            board = chess.Board(fen)
                            if moves:
                                # First move is often the opponent's move
                                # Second move is the puzzle solution
                                if len(moves) >= 2:
                                    board.push_uci(moves[0])
                                    solution_move = moves[1]
                                else:
                                    solution_move = moves[0]
                                
                                self.puzzles.append({
                                    'fen': board.fen(),
                                    'solution': solution_move,
                                    'rating': rating,
                                })
                    except Exception as e:
                        logger.debug(f"Skipping puzzle {i}: {e}")
        
        elif path.suffix == '.json':
            import json
            with open(path, 'r') as f:
                data = json.load(f)
            
            for i, puzzle in enumerate(data[:max_puzzles] if max_puzzles else data):
                self.puzzles.append(puzzle)
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get a puzzle sample.
        
        Returns:
            (board_tensor, move_index, value_target)
        """
        import chess
        
        puzzle = self.puzzles[idx]
        board = chess.Board(puzzle['fen'])
        
        # Board representation
        tensor = board_to_tensor(
            board, 
            move_number=board.fullmove_number,
            input_channels=self.input_channels
        )
        
        # Move target
        try:
            move = chess.Move.from_uci(puzzle['solution'])
            move_idx = get_move_index(move)
        except:
            move_idx = 0  # Fallback
        
        # Value target: 1.0 for white to win, -1.0 for black to win
        # Puzzles are usually from winning side's perspective
        value = 1.0 if board.turn else -1.0
        
        return (
            torch.from_numpy(tensor).float(),
            move_idx,
            value,
        )


def train_on_puzzles(
    model: nn.Module,
    puzzle_file: str,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    device: str = "cuda",
    checkpoint_dir: str = "checkpoints",
    input_channels: int = 22,
) -> Dict[str, Any]:
    """
    Train a model on puzzles (one-time bootstrap).
    
    This is supervised learning to give the model basic tactical knowledge
    before self-play begins. Should only be called ONCE per model.
    
    Args:
        model: Model to train
        puzzle_file: Path to puzzle dataset
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoint
        input_channels: Input channels for board representation
    
    Returns:
        Training statistics dict
    """
    
    logger.info("="*60)
    logger.info("PUZZLE BOOTSTRAP TRAINING")
    logger.info("="*60)
    logger.info(f"Puzzle file: {puzzle_file}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info("="*60)
    
    # Load dataset
    dataset = PuzzleDataset(puzzle_file, input_channels=input_channels)
    
    if len(dataset) == 0:
        logger.error("No puzzles loaded. Skipping bootstrap.")
        return {"error": "No puzzles loaded"}
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    # Setup training
    model.to(device)
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()
    
    # Training loop
    stats = {
        "epochs": epochs,
        "total_batches": 0,
        "epoch_losses": [],
        "final_loss": 0,
    }
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        batch_count = 0
        
        for batch_idx, (boards, move_targets, value_targets) in enumerate(dataloader):
            boards = boards.to(device)
            move_targets = move_targets.to(device)
            value_targets = value_targets.float().to(device)
            
            # Forward pass
            policy_logits, value_pred = model(boards)
            
            # Compute losses
            policy_loss = policy_loss_fn(policy_logits, move_targets)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            
            # Weighted sum (policy more important for puzzles)
            loss = 2.0 * policy_loss + 1.0 * value_loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            batch_count += 1
            stats["total_batches"] += 1
        
        avg_loss = epoch_loss / batch_count
        avg_policy = epoch_policy_loss / batch_count
        avg_value = epoch_value_loss / batch_count
        
        stats["epoch_losses"].append(avg_loss)
        
        logger.info(
            f"Epoch {epoch+1}/{epochs}: "
            f"loss={avg_loss:.4f} "
            f"(policy={avg_policy:.4f}, value={avg_value:.4f})"
        )
    
    stats["final_loss"] = stats["epoch_losses"][-1] if stats["epoch_losses"] else 0
    
    # Save checkpoint
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / "bootstrap_checkpoint.pt"
    
    torch.save({
        "state_dict": model.state_dict(),
        "bootstrap_stats": stats,
        "input_channels": input_channels,
    }, checkpoint_path)
    
    logger.info(f"Saved bootstrap checkpoint: {checkpoint_path}")
    logger.info("="*60)
    logger.info("BOOTSTRAP COMPLETE - From now on, only MCTS self-play improves the model")
    logger.info("="*60)
    
    return stats


def load_bootstrap_checkpoint(
    model: nn.Module,
    checkpoint_dir: str = "checkpoints",
    variant: str = "baseline",
) -> bool:
    """
    Load bootstrap checkpoint if it exists.
    
    Args:
        model: Model to load weights into
        checkpoint_dir: Directory with checkpoints
        variant: Model variant name
    
    Returns:
        True if checkpoint was loaded, False otherwise
    """
    
    checkpoint_path = Path(checkpoint_dir) / f"bootstrap_{variant}.pt"
    
    if not checkpoint_path.exists():
        # Try generic bootstrap checkpoint
        checkpoint_path = Path(checkpoint_dir) / "bootstrap_checkpoint.pt"
    
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            logger.info(f"Loaded bootstrap checkpoint: {checkpoint_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load bootstrap checkpoint: {e}")
    
    return False


if __name__ == "__main__":
    """
    Command-line interface for bootstrap training.
    
    Usage:
        python puzzle_train.py --puzzle-file puzzles.csv --epochs 5 --variant baseline
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Bootstrap training on puzzles")
    parser.add_argument("--puzzle-file", required=True, help="Path to puzzle CSV/JSON")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--variant", default="baseline", help="Model variant")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Create model
    input_channels = 18 if args.variant in ["baseline", "est"] else 22
    model = create_model(args.variant)
    
    # Train
    stats = train_on_puzzles(
        model=model,
        puzzle_file=args.puzzle_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        input_channels=input_channels,
    )
    
    print(f"\nBootstrap complete. Final loss: {stats['final_loss']:.4f}")
