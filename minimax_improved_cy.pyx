# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Cython-accelerated functions for Minimax chess engine.
Compile with: python setup_cython.py build_ext --inplace
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport abs as c_abs

import chess

# Type definitions
ctypedef np.int32_t INT32
ctypedef np.float64_t FLOAT64

# Piece type to index mapping (chess.PAWN=1, etc.)
cdef dict PIECE_TO_IDX = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}

# Piece values in centipawns (indexed by piece_type - 1)
cdef int[6] PIECE_VALUES = [100, 320, 330, 500, 900, 20000]


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int evaluate_material_pst_cy(object board, 
                                    np.ndarray[INT32, ndim=1] pst_white,
                                    np.ndarray[INT32, ndim=1] pst_black,
                                    np.ndarray[INT32, ndim=1] king_pst_eg,
                                    bint is_endgame):
    """
    Fast material and PST evaluation using Cython.
    
    Args:
        board: chess.Board object
        pst_white: Flattened PST array for white (64*6 elements)
        pst_black: Flattened PST array for black (64*6 elements)
        king_pst_eg: Endgame king PST (64 elements)
        is_endgame: Whether we're in endgame phase
    
    Returns:
        Score from white's perspective
    """
    cdef int score = 0
    cdef int square, piece_type, piece_idx
    cdef int piece_val, pst_val
    cdef bint is_white
    cdef object piece
    cdef dict piece_map = board.piece_map()
    
    for square, piece in piece_map.items():
        piece_type = piece.piece_type
        is_white = piece.color
        piece_idx = piece_type - 1  # Convert to 0-indexed
        
        # Material value
        piece_val = PIECE_VALUES[piece_idx]
        
        # PST value
        if piece_type == 6 and is_endgame:  # King in endgame
            if is_white:
                pst_val = king_pst_eg[square]
            else:
                pst_val = king_pst_eg[63 - square]  # Mirror for black
        else:
            if is_white:
                pst_val = pst_white[square * 6 + piece_idx]
            else:
                pst_val = pst_black[square * 6 + piece_idx]
        
        if is_white:
            score += piece_val + pst_val
        else:
            score -= piece_val + pst_val
    
    return score


@cython.boundscheck(False)
cpdef int score_move_cy(object board, object move, int ply,
                        object tt_move, list killer_moves, 
                        dict history, dict piece_value):
    """
    Fast move scoring for move ordering.
    """
    cdef int score = 0
    cdef int victim_val, attacker_val
    cdef object captured, attacker
    cdef tuple move_key
    
    # TT move highest priority
    if tt_move is not None and move == tt_move:
        return 100000
    
    # Killer moves
    if ply < len(killer_moves):
        if move == killer_moves[ply][0]:
            return 90000
        if move == killer_moves[ply][1]:
            return 89000
    
    # Captures: MVV-LVA (simple version, SEE done in Python)
    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if captured is not None:
            victim_val = piece_value[captured.piece_type]
            attacker_val = piece_value[attacker.piece_type]
            score = 50000 + victim_val * 10 - attacker_val
        else:
            score = 50000 + 1000 - 100  # En passant
        return score
    
    # Promotions
    if move.promotion:
        if move.promotion == 5:  # QUEEN
            return 60000
        return 55000
    
    # History heuristic
    move_key = (move.from_square, move.to_square)
    return history.get(move_key, 0)


@cython.boundscheck(False)
cpdef int quiescence_cy(object evaluate_fn, object board, 
                        int alpha, int beta, bint maximizing,
                        dict piece_value, int max_depth=8):
    """
    Cython-accelerated quiescence search.
    """
    cdef int stand_pat, delta, captured_val
    cdef int eval_score
    cdef object move, captured
    
    stand_pat = evaluate_fn(board)
    
    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat
    
    if max_depth <= 0:
        return stand_pat
    
    # Only search captures
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
        
        captured = board.piece_at(move.to_square)
        
        # Delta pruning
        if captured is not None:
            delta = piece_value[captured.piece_type]
        else:
            delta = 100  # En passant
        
        if maximizing:
            if stand_pat + delta + 200 < alpha:
                continue
        else:
            if stand_pat - delta - 200 > beta:
                continue
        
        board.push(move)
        eval_score = quiescence_cy(evaluate_fn, board, alpha, beta, 
                                   not maximizing, piece_value, max_depth - 1)
        board.pop()
        
        if maximizing:
            if eval_score > alpha:
                alpha = eval_score
            if alpha >= beta:
                return beta
        else:
            if eval_score < beta:
                beta = eval_score
            if beta <= alpha:
                return alpha
    
    return alpha if maximizing else beta


@cython.boundscheck(False)
cpdef tuple count_attacks_cy(object board):
    """
    Fast attack map counting using bitboards.
    Returns (white_attacks, black_attacks) as counts.
    """
    cdef int white_count = 0
    cdef int black_count = 0
    cdef object white_attacks, black_attacks
    cdef int sq
    
    # Use SquareSet for efficient bitboard operations
    white_attacks = chess.SquareSet()
    black_attacks = chess.SquareSet()
    
    # Count attacks from knights, bishops, rooks, queens
    for piece_type in [2, 3, 4, 5]:  # KNIGHT, BISHOP, ROOK, QUEEN
        for sq in board.pieces(piece_type, True):  # White
            white_attacks |= board.attacks(sq)
        for sq in board.pieces(piece_type, False):  # Black
            black_attacks |= board.attacks(sq)
    
    return len(white_attacks), len(black_attacks)
