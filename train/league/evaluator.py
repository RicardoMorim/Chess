"""
League Evaluator
================

Handles low-frequency evaluation games against frozen checkpoints.
Purpose: Catch regressions and measure strength progression.

Design principles:
- Only uses self-play with MCTS (no policy distillation)
- Tests against specific checkpoints (not training)
- Low frequency (doesn't slow down training loop)
- Provides strength trend signal
"""

import sys
import torch
import chess
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Evaluates current models against frozen opponent checkpoints.
    
    Uses deterministic MCTS (low temperature) for consistent evaluation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        eval_games_per_matchup: int = 20,
        mcts_visits: int = 400,
    ):
        """
        Initialize evaluator.
        
        Args:
            device: Device to run on
            eval_games_per_matchup: Number of games per matchup
            mcts_visits: MCTS visits per move (should be < training visits)
        """
        self.device = device
        self.eval_games_per_matchup = eval_games_per_matchup
        self.mcts_visits = mcts_visits
        
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
    
    def register_checkpoint(
        self,
        variant: str,
        step: int,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        """
        Register a checkpoint for future evaluation.
        
        Args:
            variant: Model variant name
            step: Training step at which checkpoint was taken
            checkpoint_path: Path to checkpoint on disk
        """
        key = f"{variant}_step_{step}"
        self.checkpoints[key] = {
            "variant": variant,
            "step": step,
            "path": checkpoint_path,
        }
        logger.info(f"Registered checkpoint: {key}")
    
    def evaluate_matchup(
        self,
        current_model: Any,
        current_variant: str,
        opponent_checkpoint: str,
        opponent_model: Any,
    ) -> Dict[str, Any]:
        """
        Run evaluation games between current model and opponent.
        
        Uses self-play with frozen MCTS (no training).
        
        Args:
            current_model: Current model (loaded and ready)
            current_variant: Variant name of current model
            opponent_checkpoint: Key of opponent checkpoint
            opponent_model: Loaded opponent model
        
        Returns:
            {
                "current_wins": int,
                "opponent_wins": int,
                "draws": int,
                "current_avg_length": float,
                "current_elo_diff": float,
            }
        """
        
        from core.mcts import MCTS
        from core.data import board_to_tensor
        
        if opponent_checkpoint not in self.checkpoints:
            logger.warning(f"Opponent checkpoint not found: {opponent_checkpoint}")
            return None
        
        # Create MCTS for both models (deterministic evaluation)
        current_mcts = MCTS(
            model=current_model,
            device=self.device,
            num_visits=self.mcts_visits,
            temperature=0.1,  # Low temp for deterministic play
            c_puct=4.0,
            add_noise=False,  # No noise in evaluation
        )
        
        opponent_mcts = MCTS(
            model=opponent_model,
            device=self.device,
            num_visits=self.mcts_visits,
            temperature=0.1,
            c_puct=4.0,
            add_noise=False,
        )
        
        results = {
            "current_wins": 0,
            "opponent_wins": 0,
            "draws": 0,
            "game_lengths": [],
        }
        
        # Play games: alternate colors to reduce bias
        for game_idx in range(self.eval_games_per_matchup):
            current_is_white = (game_idx % 2 == 0)
            
            game_result, length = self._play_eval_game(
                current_mcts if current_is_white else opponent_mcts,
                opponent_mcts if current_is_white else current_mcts,
                is_current_white=current_is_white,
            )
            
            results["game_lengths"].append(length)
            
            if game_result == 1:  # Current wins
                results["current_wins"] += 1
            elif game_result == -1:  # Opponent wins
                results["opponent_wins"] += 1
            else:  # Draw
                results["draws"] += 1
        
        # Compute statistics
        current_score = results["current_wins"] + 0.5 * results["draws"]
        opponent_score = results["opponent_wins"] + 0.5 * results["draws"]
        
        total_games = self.eval_games_per_matchup
        current_win_rate = current_score / total_games if total_games > 0 else 0
        
        # Simple ELO approximation
        elo_diff = self._estimate_elo_diff(current_win_rate)
        
        results["current_score"] = current_score
        results["opponent_score"] = opponent_score
        results["current_win_rate"] = current_win_rate
        results["estimated_elo_diff"] = elo_diff
        results["avg_game_length"] = (
            sum(results["game_lengths"]) / len(results["game_lengths"])
            if results["game_lengths"] else 0
        )
        
        return results
    
    def _play_eval_game(
        self,
        white_mcts: Any,
        black_mcts: Any,
        is_current_white: bool,
    ) -> Tuple[int, int]:
        """
        Play a single evaluation game.
        
        Args:
            white_mcts: MCTS for white
            black_mcts: MCTS for black
            is_current_white: Whether current model is white
        
        Returns:
            (result, game_length) where:
            - result: 1 if current wins, -1 if opponent wins, 0 if draw
            - game_length: Number of half-moves
        """
        board = chess.Board()
        move_count = 0
        max_moves = 512
        
        while not board.is_game_over() and move_count < max_moves:
            mcts = white_mcts if board.turn == chess.WHITE else black_mcts
            
            try:
                _, move = mcts.search(board)
            except Exception as e:
                logger.warning(f"MCTS search failed in evaluation: {e}")
                move = None
            
            if move is None:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = legal_moves[0]
            
            board.push(move)
            move_count += 1
        
        # Determine outcome
        outcome = board.result()
        
        if outcome == "1-0":
            result = 1 if is_current_white else -1
        elif outcome == "0-1":
            result = -1 if is_current_white else 1
        else:
            result = 0
        
        return result, move_count
    
    def _estimate_elo_diff(self, win_rate: float) -> float:
        """
        Estimate ELO difference from win rate.
        
        Uses standard formula: ELO_diff = 400 * log10(W / (1-W))
        
        Args:
            win_rate: Win rate (0 to 1)
        
        Returns:
            Estimated ELO difference
        """
        import math
        
        # Clip to avoid log(0)
        win_rate = max(0.01, min(0.99, win_rate))
        
        if win_rate == 0.5:
            return 0.0
        
        elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
        return elo_diff
    
    def get_regression_report(
        self,
        current_variant: str,
        threshold_elo_loss: float = 50,
    ) -> Dict[str, Any]:
        """
        Check if any evaluation shows regression.
        
        Args:
            current_variant: Variant to check
            threshold_elo_loss: ELO loss threshold for regression alert
        
        Returns:
            {
                "is_regression": bool,
                "worst_matchup": str,
                "worst_elo_diff": float,
            }
        """
        
        worst_elo_diff = 0
        worst_matchup = None
        
        for checkpoint_key, checkpoint_data in self.checkpoints.items():
            if checkpoint_data["variant"] != current_variant:
                continue
            
            # This would need to track historical evaluation results
            # For now, return placeholder
            
        return {
            "is_regression": False,
            "worst_matchup": worst_matchup,
            "worst_elo_diff": worst_elo_diff,
        }
