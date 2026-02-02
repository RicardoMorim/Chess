"""
Puzzle Evaluation
=================

Evaluate model's puzzle-solving accuracy.

This is a MEASUREMENT tool - it doesn't improve the model.
Use it to track tactical strength at checkpoints.
"""

import sys
import torch
import chess
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from core import create_model, board_to_tensor, get_move_index

logger = logging.getLogger(__name__)


def evaluate_puzzles(
    model: torch.nn.Module,
    puzzle_file: str,
    device: str = "cuda",
    max_puzzles: Optional[int] = 1000,
    use_mcts: bool = False,
    mcts_visits: int = 100,
    input_channels: int = 22,
) -> Dict[str, Any]:
    """
    Evaluate model accuracy on puzzles.
    
    This is MEASUREMENT ONLY - does not update model weights.
    
    Args:
        model: Model to evaluate
        puzzle_file: Path to puzzle dataset
        device: Device to run on
        max_puzzles: Maximum puzzles to evaluate
        use_mcts: Whether to use MCTS (slower but more accurate)
        mcts_visits: MCTS visits if use_mcts=True
        input_channels: Input channels for board representation
    
    Returns:
        Evaluation statistics dict
    """
    
    from bootstrap.puzzle_train import PuzzleDataset
    
    logger.info("="*60)
    logger.info("PUZZLE EVALUATION (Measurement Only)")
    logger.info("="*60)
    
    # Load puzzles
    dataset = PuzzleDataset(puzzle_file, input_channels=input_channels, max_puzzles=max_puzzles)
    
    if len(dataset) == 0:
        logger.error("No puzzles loaded")
        return {"error": "No puzzles loaded"}
    
    model.to(device)
    model.eval()
    
    # Optional MCTS
    mcts = None
    if use_mcts:
        from core import MCTS
        mcts = MCTS(
            model=model,
            device=device,
            num_visits=mcts_visits,
            temperature=0.1,  # Low temp for evaluation
            add_noise=False,
        )
    
    # Evaluate
    results = {
        "total": 0,
        "correct": 0,
        "top3_correct": 0,
        "top5_correct": 0,
        "by_rating": defaultdict(lambda: {"total": 0, "correct": 0}),
    }
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            puzzle = dataset.puzzles[idx]
            board_tensor, target_move_idx, _ = dataset[idx]
            
            # Get model's prediction
            board_tensor = board_tensor.unsqueeze(0).to(device)
            policy_logits, value = model(board_tensor)
            
            if use_mcts and mcts is not None:
                # Use MCTS for move selection
                board = chess.Board(puzzle['fen'])
                try:
                    policy, move = mcts.search(board)
                    predicted_move_idx = get_move_index(move)
                except:
                    predicted_move_idx = policy_logits.argmax(dim=1).item()
            else:
                # Use raw policy
                predicted_move_idx = policy_logits.argmax(dim=1).item()
            
            # Check correctness
            correct = (predicted_move_idx == target_move_idx)
            
            # Top-k accuracy
            topk_indices = policy_logits.topk(5, dim=1).indices.squeeze().tolist()
            top3_correct = target_move_idx in topk_indices[:3]
            top5_correct = target_move_idx in topk_indices[:5]
            
            # Update stats
            results["total"] += 1
            results["correct"] += int(correct)
            results["top3_correct"] += int(top3_correct)
            results["top5_correct"] += int(top5_correct)
            
            # By rating bucket
            rating = puzzle.get('rating', 1500)
            bucket = (rating // 200) * 200  # 1200, 1400, 1600, etc.
            results["by_rating"][bucket]["total"] += 1
            results["by_rating"][bucket]["correct"] += int(correct)
            
            if (idx + 1) % 100 == 0:
                acc = results["correct"] / results["total"] * 100
                logger.info(f"Evaluated {idx+1}/{len(dataset)}: accuracy={acc:.1f}%")
    
    # Compute final stats
    results["accuracy"] = results["correct"] / results["total"] * 100
    results["top3_accuracy"] = results["top3_correct"] / results["total"] * 100
    results["top5_accuracy"] = results["top5_correct"] / results["total"] * 100
    
    # Convert by_rating to regular dict
    results["by_rating"] = dict(results["by_rating"])
    for bucket in results["by_rating"]:
        bucket_data = results["by_rating"][bucket]
        bucket_data["accuracy"] = bucket_data["correct"] / bucket_data["total"] * 100
    
    # Log summary
    logger.info("="*60)
    logger.info("PUZZLE EVALUATION RESULTS")
    logger.info("="*60)
    logger.info(f"Total puzzles: {results['total']}")
    logger.info(f"Top-1 accuracy: {results['accuracy']:.1f}%")
    logger.info(f"Top-3 accuracy: {results['top3_accuracy']:.1f}%")
    logger.info(f"Top-5 accuracy: {results['top5_accuracy']:.1f}%")
    logger.info("")
    logger.info("By rating:")
    for bucket in sorted(results["by_rating"].keys()):
        bucket_data = results["by_rating"][bucket]
        logger.info(f"  {bucket}-{bucket+199}: {bucket_data['accuracy']:.1f}% ({bucket_data['correct']}/{bucket_data['total']})")
    logger.info("="*60)
    
    return results


def compare_checkpoints(
    checkpoint_paths: List[str],
    puzzle_file: str,
    variant: str = "baseline",
    device: str = "cuda",
    max_puzzles: int = 500,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare puzzle accuracy across multiple checkpoints.
    
    Useful for tracking improvement over training.
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        puzzle_file: Path to puzzle dataset
        variant: Model variant
        device: Device to run on
        max_puzzles: Max puzzles per evaluation
    
    Returns:
        Dict mapping checkpoint_path -> evaluation results
    """
    
    results = {}
    
    for ckpt_path in checkpoint_paths:
        logger.info(f"\nEvaluating: {ckpt_path}")
        
        # Load model
        model = create_model(variant)
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Evaluate
        input_channels = 18 if variant in ["baseline", "est"] else 22
        eval_results = evaluate_puzzles(
            model=model,
            puzzle_file=puzzle_file,
            device=device,
            max_puzzles=max_puzzles,
            input_channels=input_channels,
        )
        
        results[ckpt_path] = eval_results
    
    # Summary comparison
    logger.info("\n" + "="*60)
    logger.info("CHECKPOINT COMPARISON")
    logger.info("="*60)
    for ckpt_path, result in results.items():
        name = Path(ckpt_path).stem
        logger.info(f"{name}: {result.get('accuracy', 0):.1f}%")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate puzzle accuracy")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--puzzle-file", required=True, help="Puzzle CSV/JSON")
    parser.add_argument("--variant", default="baseline", help="Model variant")
    parser.add_argument("--max-puzzles", type=int, default=1000, help="Max puzzles")
    parser.add_argument("--use-mcts", action="store_true", help="Use MCTS")
    parser.add_argument("--device", default="cuda", help="Device")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    # Load model
    model = create_model(args.variant)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    # Evaluate
    input_channels = 18 if args.variant in ["baseline", "est"] else 22
    results = evaluate_puzzles(
        model=model,
        puzzle_file=args.puzzle_file,
        device=args.device,
        max_puzzles=args.max_puzzles,
        use_mcts=args.use_mcts,
        input_channels=input_channels,
    )
    
    print(f"\nFinal accuracy: {results['accuracy']:.1f}%")
