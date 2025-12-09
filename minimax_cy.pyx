"""
Lightweight Cython helpers for Minimax.

These mirror the Python implementations in `Minimax.py` so we can get a speed
boost from compiled loops while keeping behavior identical. Signatures are
kept the same as the calls in `Minimax.py`:

- score_move_cy(board, move, ply, tt_move, killer_moves, history, piece_value)
- evaluate_static_cy(piece_value, pst_white_flat, pst_black_flat, board)
- quiescence_cy(evaluate_fn, board, alpha, beta, maximizing_player, piece_value)

All arguments are regular Python objects; we rely on Cython to remove some of
the Python-level overhead in tight loops. This is intentionally minimal and
compatible with the current Minimax implementation.
"""

import cython
import chess


@cython.boundscheck(False)
@cython.wraparound(False)
def score_move_cy(board, move, int ply, tt_move,
                  killer_moves, history, piece_value):
    """Move ordering score (mirrors Python version)."""
    if tt_move is not None and move == tt_move:
        return 20000

    if ply < len(killer_moves):
        killers = killer_moves[ply]
        if move in killers:
            return 10000

    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)
        if captured_piece is not None and attacker_piece is not None:
            victim_value = piece_value[captured_piece.piece_type]
            attacker_value = piece_value[attacker_piece.piece_type]
            return 9000 + (victim_value * 10 - attacker_value)
        else:
            return 8000  # en passant or missing piece safety

    if move.promotion:
        return 9500 if move.promotion == chess.QUEEN else 8500

    if board.gives_check(move):
        return 8000

    move_key = (move.from_square, move.to_square)
    return history.get(move_key, 0)


@cython.boundscheck(False)
@cython.wraparound(False)
def evaluate_static_cy(piece_value, pst_white_flat, pst_black_flat, board, king_safety_fn=None):
    """Static evaluation using flattened PSTs (square-major, piece-index).

    If king_safety_fn is provided, it is called to keep parity with the
    Python implementation.
    """
    cdef long material = 0
    cdef long pst_score = 0
    cdef long king_safety = 0

    for square, piece in board.piece_map().items():
        piece_idx = piece.piece_type  # 1..6 in python-chess
        piece_idx -= 1                # convert to 0-based index
        if piece.color == chess.WHITE:
            material += piece_value[piece.piece_type]
            pst_score += pst_white_flat[square * 6 + piece_idx]
        else:
            material -= piece_value[piece.piece_type]
            pst_score -= pst_black_flat[square * 6 + piece_idx]

    if king_safety_fn is not None:
        king_safety = king_safety_fn(board)

    return material + pst_score + king_safety


@cython.boundscheck(False)
@cython.wraparound(False)
def quiescence_cy(evaluate_fn, board, alpha, beta, bint maximizing_player,
                 piece_value):
    """Quiescence search using the provided evaluate_fn (Python callable)."""
    stand_pat = evaluate_fn(board)
    if maximizing_player:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        captured_piece = board.piece_at(move.to_square)
        delta = piece_value[captured_piece.piece_type] if captured_piece else 100
        if maximizing_player and stand_pat + delta + 200 < alpha:
            continue
        if (not maximizing_player) and stand_pat - delta - 200 > beta:
            continue

        board.push(move)
        score = quiescence_cy(evaluate_fn, board, alpha, beta, not maximizing_player, piece_value)
        board.pop()

        if maximizing_player:
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break
        else:
            if score < beta:
                beta = score
            if beta <= alpha:
                break

    return alpha if maximizing_player else beta
