import unittest

import chess

from train.league.evaluator import Evaluator
from train.league.fair_evaluation import build_fair_opening_fens, round_robin_pairs


class FairEvaluationTests(unittest.TestCase):
    def test_build_fair_opening_fens(self):
        fens = build_fair_opening_fens()

        self.assertGreaterEqual(len(fens), 1)
        self.assertEqual(len(fens), len(set(fens)))
        for fen in fens:
            board = chess.Board(fen)
            self.assertTrue(board.is_valid())

    def test_round_robin_pairs(self):
        self.assertEqual(
            round_robin_pairs(["baseline", "attack", "est"]),
            [("baseline", "attack"), ("baseline", "est"), ("attack", "est")],
        )

    def test_compare_all_variants_aggregates_pairwise_results(self):
        evaluator = Evaluator(device="cpu", eval_games_per_matchup=2, mcts_visits=1)

        calls = []

        def fake_evaluate_pair(**kwargs):
            current = kwargs["current_variant"]
            opponent = kwargs["opponent_variant"]
            calls.append((current, opponent))
            return {
                "current_variant": current,
                "opponent_variant": opponent,
                "current_score": 1.5,
                "opponent_score": 0.5,
                "current_wins": 1,
                "opponent_wins": 0,
                "draws": 1,
                "total_games": 2,
                "current_win_rate": 0.75,
                "estimated_elo_diff": 190.0,
            }

        evaluator.evaluate_pair = fake_evaluate_pair  # type: ignore[assignment]
        result = evaluator.compare_all_variants(
            {"baseline": object(), "attack": object(), "est": object()}
        )

        self.assertEqual(calls, [("baseline", "attack"), ("baseline", "est"), ("attack", "est")])
        self.assertEqual(result["opening_suite_size"], len(build_fair_opening_fens()))
        self.assertAlmostEqual(result["scoreboard"]["baseline"]["avg_score"], 0.75)
        self.assertAlmostEqual(result["scoreboard"]["attack"]["avg_score"], 0.5)
        self.assertAlmostEqual(result["scoreboard"]["est"]["avg_score"], 0.25)


if __name__ == "__main__":
    unittest.main()
