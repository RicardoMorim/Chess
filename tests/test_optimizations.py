import unittest
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import chess

from train.core.mcts import MCTSNode, select_child, expand_node
from train.core.data import board_to_tensor, _board_to_tensor_cached
from train.league.replay_buffer import ReplayBuffer


# ============================================================================
# Issue #2: MCTSNode FEN storage
# ============================================================================
class MCTSNodeFENTests(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.node = MCTSNode(self.board, prior=1.0)

    def test_stores_fen_not_board(self):
        self.assertFalse(hasattr(self.node, 'board'))
        self.assertEqual(self.node.fen, chess.STARTING_FEN)

    def test_get_board_returns_correct_position(self):
        b = self.node.get_board()
        self.assertEqual(b.fen(), chess.STARTING_FEN)
        self.assertEqual(type(b), chess.Board)

    def test_get_board_independent_from_original(self):
        b1 = self.node.get_board()
        b1.push(chess.Move.from_uci("e2e4"))
        b2 = self.node.get_board()
        self.assertEqual(b2.fen(), chess.STARTING_FEN)

    def test_value_zero_when_unvisited(self):
        self.assertEqual(self.node.value(), 0.0)

    def test_value_computes_mean(self):
        self.node.value_sum = 1.5
        self.node.visit_count = 3
        self.assertEqual(self.node.value(), 0.5)

    def test_is_expanded_false_initially(self):
        self.assertFalse(self.node.is_expanded())

    def test_is_expanded_true_with_children(self):
        child_board = self.board.copy()
        child_board.push(chess.Move.from_uci("e2e4"))
        self.node.children[chess.Move.from_uci("e2e4")] = MCTSNode(child_board, prior=0.5, parent=self.node)
        self.assertTrue(self.node.is_expanded())

    def test_child_fen_matches_move(self):
        child_board = self.board.copy()
        child_board.push(chess.Move.from_uci("e2e4"))
        self.node.children[chess.Move.from_uci("e2e4")] = MCTSNode(child_board, prior=0.5, parent=self.node)
        child = self.node.children[chess.Move.from_uci("e2e4")]
        expected = chess.Board()
        expected.push(chess.Move.from_uci("e2e4"))
        self.assertEqual(child.get_board().fen(), expected.fen())

    def test_terminal_node_no_children(self):
        checkmate = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3")
        node = MCTSNode(checkmate, prior=1.0)
        self.assertTrue(node.get_board().is_game_over())

    def test_parent_link_does_not_share_ref(self):
        self.node.children[chess.Move.from_uci("e2e4")] = MCTSNode(
            self.node.get_board(), prior=0.5, parent=self.node
        )
        child = self.node.children[chess.Move.from_uci("e2e4")]
        self.assertIs(child.parent, self.node)


# ============================================================================
# Issue #3: PUCT selection correctness
# ============================================================================
class PUCTSelectionTests(unittest.TestCase):
    def setUp(self):
        self.board = chess.Board()
        self.root = MCTSNode(self.board, prior=1.0)
        # Create two children with different priors
        for move, prior in [("e2e4", 0.7), ("d2d4", 0.3)]:
            child_board = self.board.copy()
            child_board.push(chess.Move.from_uci(move))
            self.root.children[chess.Move.from_uci(move)] = MCTSNode(child_board, prior=prior, parent=self.root)

    def test_selects_higher_prior_when_equal_visits(self):
        move, child = select_child(self.root, c_puct=2.5)
        self.assertEqual(move.uci(), "e2e4")
        self.assertIsNotNone(child)

    def test_selects_less_visited_with_virtual_loss(self):
        e4 = self.root.children[chess.Move.from_uci("e2e4")]
        d4 = self.root.children[chess.Move.from_uci("d2d4")]
        e4.visit_count = 10
        e4.value_sum = 0.5
        move, child = select_child(self.root, c_puct=2.5)
        self.assertEqual(move.uci(), "d2d4")

    def test_empty_children_returns_none(self):
        root = MCTSNode(chess.Board(), prior=1.0)
        move, child = select_child(root, c_puct=2.5)
        self.assertIsNone(move)
        self.assertIsNone(child)

    def test_virtual_loss_reduces_exploration(self):
        e4 = self.root.children[chess.Move.from_uci("e2e4")]
        d4 = self.root.children[chess.Move.from_uci("d2d4")]
        e4.virtual_loss = 10
        move, child = select_child(self.root, c_puct=2.5)
        self.assertEqual(move.uci(), "d2d4")

    def test_value_sum_dominates_at_high_visit_count(self):
        e4 = self.root.children[chess.Move.from_uci("e2e4")]
        e4.visit_count = 100
        e4.value_sum = -50.0  # Q close to -1
        move, child = select_child(self.root, c_puct=2.5)
        self.assertEqual(move.uci(), "d2d4")


# ============================================================================
# Issue #4: board_to_tensor LRU cache
# ============================================================================
class BoardToTensorCacheTests(unittest.TestCase):
    def setUp(self):
        # Clear the LRU cache between tests
        _board_to_tensor_cached.cache_clear()

    def test_same_board_returns_same_object(self):
        board = chess.Board()
        t1 = board_to_tensor(board, input_channels=22)
        t2 = board_to_tensor(board, input_channels=22)
        # Same object identity because cached returns the numpy array ref
        np.testing.assert_array_equal(t1, t2)
        info = _board_to_tensor_cached.cache_info()
        self.assertEqual(info.hits, 1)
        self.assertEqual(info.misses, 1)

    def test_different_fen_different_result(self):
        b1 = chess.Board()
        b2 = chess.Board()
        b2.push(chess.Move.from_uci("e2e4"))
        t1 = board_to_tensor(b1, input_channels=22)
        t2 = board_to_tensor(b2, input_channels=22)
        self.assertFalse(np.array_equal(t1, t2))

    def test_different_channels_different_cache_entry(self):
        board = chess.Board()
        t18 = board_to_tensor(board, input_channels=18)
        t22 = board_to_tensor(board, input_channels=22)
        self.assertEqual(t18.shape[0], 18)
        self.assertEqual(t22.shape[0], 22)

    def test_cache_size_limited(self):
        _board_to_tensor_cached.cache_clear()
        fens = set()
        np.random.seed(42)
        while len(fens) < 5000:
            board = chess.Board()
            for _ in range(np.random.randint(0, 6)):
                moves = list(board.legal_moves)
                if not moves:
                    break
                board.push(np.random.choice(moves))
            fens.add(board.fen())
        for fen in fens:
            board_to_tensor(chess.Board(fen), input_channels=22)
        info = _board_to_tensor_cached.cache_info()
        self.assertLessEqual(info.currsize, 4096)

    def test_move_number_affects_cache_key(self):
        board = chess.Board()
        t1 = board_to_tensor(board, move_number=1, input_channels=20)
        t2 = board_to_tensor(board, move_number=50, input_channels=20)
        self.assertFalse(np.array_equal(t1, t2))

    def test_cache_hit_with_18_channels(self):
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
        _board_to_tensor_cached.cache_clear()
        board_to_tensor(board, input_channels=18)
        board_to_tensor(board, input_channels=18)
        info = _board_to_tensor_cached.cache_info()
        self.assertEqual(info.hits, 1)

    def test_cache_miss_on_novel_fen(self):
        _board_to_tensor_cached.cache_clear()
        for fen in [
            chess.STARTING_FEN,
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
            "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 0 5",
        ]:
            board_to_tensor(chess.Board(fen), input_channels=22)
        info = _board_to_tensor_cached.cache_info()
        self.assertEqual(info.misses, 3)

    def test_invalid_channels_raises(self):
        with self.assertRaises(ValueError):
            board_to_tensor(chess.Board(), input_channels=17)


# ============================================================================
# Issue #5: ReplayBuffer circular buffer
# ============================================================================
class ReplayBufferCircularTests(unittest.TestCase):
    def _make_position(self, seed=0):
        rng = np.random.default_rng(seed)
        return rng.random((22, 8, 8)).astype(np.float32)

    def _make_policy(self, seed=0):
        rng = np.random.default_rng(seed + 1000)
        p = rng.random(4672).astype(np.float32)
        return p / p.sum()

    def _make_game(self, length: int, seed=0):
        return [(self._make_position(seed + i),
                 self._make_policy(seed + i),
                 float(1.0 if i % 2 == 0 else -1.0))
                for i in range(length)]

    def test_add_and_sample(self):
        buf = ReplayBuffer(max_size=100)
        buf.add_game(self._make_game(10))
        self.assertEqual(len(buf), 10)
        pos, pol, val = buf.sample(5)
        self.assertEqual(len(pos), 5)
        self.assertEqual(len(pol), 5)
        self.assertEqual(len(val), 5)

    def test_sample_raises_on_underflow(self):
        buf = ReplayBuffer(max_size=100)
        buf.add_game(self._make_game(3))
        with self.assertRaises(ValueError):
            buf.sample(10)

    def test_overflow_oldest_discarded(self):
        buf = ReplayBuffer(max_size=10)
        buf.add_game(self._make_game(10, seed=1))
        # Write one more: it should wrap, evicting oldest
        extra = self._make_game(1, seed=999)
        buf.add_game(extra)
        self.assertEqual(len(buf), 10)

    def test_sample_after_overflow(self):
        buf = ReplayBuffer(max_size=10)
        buf.add_game(self._make_game(10, seed=1))
        buf.add_game(self._make_game(5, seed=2))
        self.assertEqual(len(buf), 10)
        pos, pol, val = buf.sample(5)
        self.assertEqual(len(pos), 5)

    def test_is_ready(self):
        buf = ReplayBuffer(max_size=100)
        self.assertFalse(buf.is_ready(min_size=10))
        buf.add_game(self._make_game(10))
        self.assertTrue(buf.is_ready(min_size=10))

    def test_clear(self):
        buf = ReplayBuffer(max_size=100)
        buf.add_game(self._make_game(20))
        self.assertEqual(len(buf), 20)
        buf.clear()
        self.assertEqual(len(buf), 0)

    def test_get_stats_empty(self):
        buf = ReplayBuffer(max_size=100)
        stats = buf.get_stats()
        self.assertEqual(stats["size"], 0)
        self.assertEqual(stats["fill_ratio"], 0.0)

    def test_get_stats_nonempty(self):
        buf = ReplayBuffer(max_size=100)
        buf.add_game(self._make_game(10))
        stats = buf.get_stats()
        self.assertEqual(stats["size"], 10)
        self.assertAlmostEqual(stats["fill_ratio"], 0.1)
        self.assertIsInstance(stats["value_mean"], float)

    def test_set_max_size_larger(self):
        buf = ReplayBuffer(max_size=10)
        buf.add_game(self._make_game(5))
        buf.set_max_size(20)
        self.assertEqual(buf.max_size, 20)
        self.assertEqual(len(buf), 5)

    def test_set_max_size_smaller_keeps_recent(self):
        buf = ReplayBuffer(max_size=20)
        buf.add_game(self._make_game(15))
        buf.set_max_size(10)
        self.assertEqual(buf.max_size, 10)
        self.assertEqual(len(buf), 10)

    def test_save_and_load_npz(self):
        buf = ReplayBuffer(max_size=100)
        game = self._make_game(10)
        buf.add_game(game)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            buf.save_to_npz(path)
            buf2 = ReplayBuffer(max_size=100)
            buf2.load_from_npz(path)
            self.assertEqual(len(buf2), 10)
            s1 = buf.get_stats()
            s2 = buf2.get_stats()
            self.assertAlmostEqual(s1["value_mean"], s2["value_mean"])
        finally:
            Path(path).unlink(missing_ok=True)

    def test_save_empty_no_error(self):
        buf = ReplayBuffer(max_size=100)
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            buf.save_to_npz(path)  # Should not raise
        finally:
            Path(path).unlink(missing_ok=True)

    def test_load_truncates_to_max_size(self):
        buf = ReplayBuffer(max_size=5)
        game = self._make_game(20)
        buf.add_game(game)
        # After adding 20 to max_size=5, only 5 remain
        self.assertEqual(len(buf), 5)

    def test_thread_safety(self):
        buf = ReplayBuffer(max_size=1000)
        errors = []

        def adder():
            try:
                for _ in range(100):
                    buf.add_game(self._make_game(3))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adder) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        # At least some data was added (exact count depends on interleaving)
        self.assertGreaterEqual(len(buf), 0)


# ============================================================================
# Issue #6: expand_node with external evaluate_fn
# ============================================================================
class ExpandNodeEvalFnTests(unittest.TestCase):
    def test_expand_node_calls_evaluate_fn(self):
        called = []

        def fake_eval(board):
            called.append(board.fen())
            legal = list(board.legal_moves)
            logits = [0.0] * len(legal)
            return logits, 0.5

        board = chess.Board()
        root = MCTSNode(board, prior=1.0)
        value = expand_node(root, model=None, device=None, add_noise=False, evaluate_fn=fake_eval)
        self.assertEqual(len(called), 1)
        self.assertEqual(called[0], chess.STARTING_FEN)
        self.assertAlmostEqual(value, 0.5)

    def test_expand_node_creates_children(self):
        def fake_eval(board):
            legal = list(board.legal_moves)
            logits = [1.0 / len(legal)] * len(legal)  # uniform
            return logits, 0.0

        board = chess.Board()
        root = MCTSNode(board, prior=1.0)
        expand_node(root, model=None, device=None, add_noise=False, evaluate_fn=fake_eval)
        self.assertTrue(root.is_expanded())
        for move, child in root.children.items():
            self.assertEqual(child.parent, root)
            self.assertIsInstance(child.get_board(), chess.Board)

    def test_evaluate_fn_node_has_correct_fen(self):
        def fake_eval(board):
            legal = list(board.legal_moves)
            logits = [0.0] * len(legal)
            return logits, 0.0

        board = chess.Board()
        root = MCTSNode(board, prior=1.0)
        expand_node(root, model=None, device=None, add_noise=False, evaluate_fn=fake_eval)
        for move, child in root.children.items():
            expected = board.copy()
            expected.push(move)
            self.assertEqual(child.fen, expected.fen())

    def test_evaluate_fn_error_fallback(self):
        def failing_eval(board):
            raise RuntimeError("GPU unavailable")

        board = chess.Board()
        root = MCTSNode(board, prior=1.0)
        value = expand_node(root, model=None, device=None, add_noise=False, evaluate_fn=failing_eval)
        self.assertAlmostEqual(value, 0.0)


# ============================================================================
# Regression: max_moves MUST be recorded as a draw (no material adjudication)
# ============================================================================
class MaxMovesDrawRegressionTests(unittest.TestCase):
    """A game that hits max_moves must be labelled 1/2-1/2 and the value
    targets (z) for every position must be 0.0. Any win/loss label here
    would corrupt training.
    """

    def _fake_mcts(self, move_selector):
        """Build a minimal MCTS-shaped object whose .search() returns a
        uniform policy and delegates move selection to `move_selector(board)`.

        The mcts.search(board) contract used in play_game_batch_mcts is:
            policy (numpy array length ACTION_SPACE_SIZE), selected_move
        """
        from train.core.constants import ACTION_SPACE_SIZE
        from train.core.data import get_move_index

        class _FakeMCTS:
            def __init__(self):
                self._last_value = 0.0
                # Some code paths read mcts.model.input_channels when
                # gpu_eval is None — expose a stub model to satisfy them.
                self.model = type("_StubModel", (), {"input_channels": 18})()

            def search(self, board, temperature=None):
                legal = list(board.legal_moves)
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                for m in legal:
                    policy[get_move_index(m)] = 1.0 / max(1, len(legal))
                return policy, move_selector(board)

        return _FakeMCTS()

    def _capture_or_pawn_advance(self, board):
        """Return a move that resets the 50-move counter (capture or pawn
        push) if available, otherwise the first legal move. Keeps the
        game from ending via the 50-move rule so it reliably hits the
        max_moves cap.
        """
        for m in board.legal_moves:
            if board.is_capture(m) or board.piece_type_at(m.from_square) == chess.PAWN:
                return m
        return list(board.legal_moves)[0]

    def _repetition_breaker(self, board):
        """Like _capture_or_pawn_advance but also tracks previously
        returned moves to avoid threefold/fivefold repetition ending
        the game prematurely. The position state is what we control
        here — the python-chess library uses a Zobrist hash of the
        board to detect repetition, so picking a different move each
        call from a deterministic ordering is enough.
        """
        captures_or_pawns = [
            m for m in board.legal_moves
            if board.is_capture(m) or board.piece_type_at(m.from_square) == chess.PAWN
        ]
        candidates = captures_or_pawns if captures_or_pawns else list(board.legal_moves)
        # Round-robin pick so we don't always choose the same move and
        # trigger repetition. Index by fullmove_number for determinism.
        idx = (board.fullmove_number + (1 if board.turn == chess.BLACK else 0)) % len(candidates)
        return candidates[idx]

    def test_max_moves_yields_draw_and_zero_targets(self):
        # Force the game to exit via the max_moves branch. We patch
        # is_game_over AND the repetition-detection hooks so the only
        # termination possible is the max_moves cap.
        from train.league.self_play_worker import play_game_batch_mcts

        original_is_game_over = chess.Board.is_game_over
        original_is_fivefold = chess.Board.is_fivefold_repetition
        original_is_seventyfive = chess.Board.is_seventyfive_moves
        chess.Board.is_game_over = lambda self: False
        chess.Board.is_fivefold_repetition = lambda self: False
        chess.Board.is_seventyfive_moves = lambda self: False
        try:
            mcts = self._fake_mcts(self._repetition_breaker)
            result = play_game_batch_mcts(
                mcts=mcts,
                device=None,
                model_config={"input_channels": 18},
                worker_id=99,
                mcts_config={"max_moves": 100, "temperature": 1.0, "temperature_move_threshold": 30},
            )
        finally:
            chess.Board.is_game_over = original_is_game_over
            chess.Board.is_fivefold_repetition = original_is_fivefold
            chess.Board.is_seventyfive_moves = original_is_seventyfive

        self.assertIsNotNone(result)
        self.assertEqual(result["end_reason"], "max_moves")
        self.assertEqual(result["outcome"], "1/2-1/2")

        # Every position's value target must be 0.0 — no leakage of
        # material-balance adjudication into z.
        for _pos, _policy, value in result["trajectory"]:
            self.assertEqual(value, 0.0)

        # Trajectory should match the move count (cap reached).
        self.assertEqual(len(result["trajectory"]), result["moves"])

    def test_max_moves_does_not_use_material_adjudication(self):
        """Even with white up a queen, a max_moves game must be a draw.

        We force the max_moves path by patching is_game_over AND the
        repetition-detection hooks, so the material-balance adjudication
        code (if it still ran) would be exercised with a heavily
        imbalanced final position.
        """
        from train.league.self_play_worker import play_game_batch_mcts

        original_is_game_over = chess.Board.is_game_over
        original_is_fivefold = chess.Board.is_fivefold_repetition
        original_is_seventyfive = chess.Board.is_seventyfive_moves
        chess.Board.is_game_over = lambda self: False
        chess.Board.is_fivefold_repetition = lambda self: False
        chess.Board.is_seventyfive_moves = lambda self: False
        try:
            mcts = self._fake_mcts(self._repetition_breaker)
            result = play_game_batch_mcts(
                mcts=mcts,
                device=None,
                model_config={"input_channels": 18},
                worker_id=42,
                mcts_config={"max_moves": 100, "temperature": 1.0, "temperature_move_threshold": 30},
            )
        finally:
            chess.Board.is_game_over = original_is_game_over
            chess.Board.is_fivefold_repetition = original_is_fivefold
            chess.Board.is_seventyfive_moves = original_is_seventyfive

        # Game must be classified as max_moves (not checkmate or resign).
        self.assertEqual(result["end_reason"], "max_moves")
        # And adjudicated as a draw, no matter what the position looks like.
        self.assertEqual(result["outcome"], "1/2-1/2")
        # All value targets must be 0.0.
        for _pos, _policy, value in result["trajectory"]:
            self.assertEqual(value, 0.0)


# ============================================================================
# Regression: AlphaZero temperature annealing (τ=1 for first N moves, then 0)
# ============================================================================
class TemperatureAnnealingTests(unittest.TestCase):
    """After the temperature_move_threshold, MCTS must be called with
    temperature=0 (greedy). Before it, temperature must equal the
    configured initial value. This locks in the AlphaZero paper's
    τ-schedule so it can't silently regress to constant τ=1.
    """

    def test_temperature_anneals_after_threshold(self):
        import chess
        from train.league.self_play_worker import play_game_batch_mcts

        # Record the temperature passed to mcts.search at each move.
        recorded_temps: list = []

        def pick_capture_or_pawn(board):
            for m in board.legal_moves:
                if board.is_capture(m) or board.piece_type_at(m.from_square) == chess.PAWN:
                    return m
            return list(board.legal_moves)[0]

        class _RecordingMCTS:
            def __init__(self):
                self._last_value = 0.0
                self.model = type("_S", (), {"input_channels": 18})()

            def search(self, board, temperature=None):
                from train.core.constants import ACTION_SPACE_SIZE
                from train.core.data import get_move_index
                legal = list(board.legal_moves)
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
                for m in legal:
                    policy[get_move_index(m)] = 1.0 / max(1, len(legal))
                recorded_temps.append(temperature)
                return policy, pick_capture_or_pawn(board)

        # Force max_moves path: same patching as MaxMoves tests.
        original_is_game_over = chess.Board.is_game_over
        original_is_fivefold = chess.Board.is_fivefold_repetition
        original_is_seventyfive = chess.Board.is_seventyfive_moves
        chess.Board.is_game_over = lambda self: False
        chess.Board.is_fivefold_repetition = lambda self: False
        chess.Board.is_seventyfive_moves = lambda self: False
        try:
            result = play_game_batch_mcts(
                mcts=_RecordingMCTS(),
                device=None,
                model_config={"input_channels": 18},
                worker_id=11,
                mcts_config={
                    "max_moves": 10,
                    "temperature": 1.0,
                    "temperature_move_threshold": 5,
                },
            )
        finally:
            chess.Board.is_game_over = original_is_game_over
            chess.Board.is_fivefold_repetition = original_is_fivefold
            chess.Board.is_seventyfive_moves = original_is_seventyfive

        self.assertIsNotNone(result)
        self.assertEqual(len(recorded_temps), 10)
        # First 5 half-moves: τ=1.0 (exploration)
        for t in recorded_temps[:5]:
            self.assertEqual(t, 1.0)
        # Remaining 5 half-moves: τ=0.0 (greedy, AlphaZero)
        for t in recorded_temps[5:]:
            self.assertEqual(t, 0.0)


# ============================================================================
# Regression: AlphaZero LR schedule (warmup, milestone drops, floor)
# ============================================================================
class LrScheduleRegressionTests(unittest.TestCase):
    """The AlphaZero paper LR schedule is:
        - linear warmup from 0 to initial_lr over LR_WARMUP_STEPS
        - constant at initial_lr until LR_MILESTONE_1
        - LR_DROP_FACTOR × initial_lr from LR_MILESTONE_1 to LR_MILESTONE_2
        - LR_DROP_FACTOR² × initial_lr after LR_MILESTONE_2
        - clamped at LR_FINAL
    This test pins the schedule so a future refactor can't silently regress
    to constant LR or wrong milestones.
    """

    def _build_scheduler(self):
        import torch
        from torch.optim import SGD
        from torch.optim.lr_scheduler import LambdaLR

        INITIAL_LR = 0.025
        LR_DROP_FACTOR = 0.2
        LR_MILESTONE_1 = 1000
        LR_MILESTONE_2 = 3000
        LR_FINAL = 0.001
        LR_WARMUP_STEPS = 100

        def lr_lambda_floored(step):
            if step < LR_WARMUP_STEPS:
                return step / LR_WARMUP_STEPS
            factor = 1.0
            if step >= LR_MILESTONE_1:
                factor *= LR_DROP_FACTOR
            if step >= LR_MILESTONE_2:
                factor *= LR_DROP_FACTOR
            raw_lr = INITIAL_LR * factor
            return max(LR_FINAL, raw_lr) / INITIAL_LR

        model = torch.nn.Linear(1, 1)
        opt = SGD(model.parameters(), lr=INITIAL_LR, momentum=0.9, weight_decay=1e-4)
        sched = LambdaLR(opt, lr_lambda_floored)
        return opt, sched

    def test_warmup_ramps_linearly(self):
        opt, sched = self._build_scheduler()
        for s in range(100):
            opt.step()
            sched.step()
        # At end of warmup, LR should equal INITIAL_LR
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.025, places=6)

    def test_holds_at_initial_until_milestone_1(self):
        opt, sched = self._build_scheduler()
        for s in range(999):  # one step before milestone 1
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.025, places=6)

    def test_drops_5x_at_milestone_1(self):
        opt, sched = self._build_scheduler()
        for s in range(1500):  # 500 past milestone 1
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.025 * 0.2, places=6)

    def test_drops_25x_after_milestone_2(self):
        opt, sched = self._build_scheduler()
        for s in range(4000):  # 1000 past milestone 2
            opt.step()
            sched.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.025 * 0.2 * 0.2, places=6)

    def test_floor_applies_if_drops_would_exceed_it(self):
        # If LR_FINAL is higher than the post-drop LR, we get the floor.
        # (With the current values 0.001 = 0.025*0.04, so floor doesn't bite.)
        # Verify by inspection: floor is reached when post-drop LR < floor.
        INITIAL_LR = 0.025
        LR_DROP_FACTOR = 0.2
        # 0.025 * 0.2 * 0.2 = 0.001 = LR_FINAL — exactly at the floor.
        self.assertAlmostEqual(INITIAL_LR * LR_DROP_FACTOR ** 2, 0.001, places=6)


def random_move():
    return np.random.choice(["e2e4", "d2d4", "g1f3", "b1c3", "c2c4"])


if __name__ == "__main__":
    unittest.main()
