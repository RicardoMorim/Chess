# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import chess
import chess.polyglot
import numpy as np
cimport numpy as np
cimport cython
from libc.stdlib cimport malloc, free

# Define types for better performance
ctypedef int piece_t
ctypedef int square_t
ctypedef int color_t
ctypedef int score_t

# Fast piece-square table lookup
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef score_t evaluate_static_cy(dict piece_value, 
                                 np.ndarray[np.int32_t, ndim=1] pst_white,
                                 np.ndarray[np.int32_t, ndim=1] pst_black,
                                 board):
    """
    Highly optimized static evaluation function using Cython.
    """
    cdef:
        score_t material = 0
        score_t pst_score = 0
        square_t square
        piece_t piece_type
        color_t color
        score_t king_safety = 0
        score_t piece_val
    
    # Process all pieces in one efficient loop
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type
        color = piece.color
        piece_val = piece_value[piece_type]
        
        if color == chess.WHITE:
            material += piece_val
            pst_score += pst_white[square * 6 + piece_type - 1]
        else:
            material -= piece_val
            pst_score += pst_black[square * 6 + piece_type - 1]
    
    # Simplified king safety evaluation
    if board.has_kingside_castling_rights(chess.WHITE):
        king_safety += 50
    if board.has_queenside_castling_rights(chess.WHITE):
        king_safety += 50
    if board.has_kingside_castling_rights(chess.BLACK):
        king_safety -= 50
    if board.has_queenside_castling_rights(chess.BLACK):
        king_safety -= 50
    
    # Return combined score
    return material + pst_score + king_safety

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef score_t quiescence_cy(evaluate_static_func, 
                          board, 
                          score_t alpha, 
                          score_t beta, 
                          bint maximizing_player,
                          dict piece_value):
    """
    Optimized quiescence search implementation.
    """
    cdef:
        score_t stand_pat = evaluate_static_func(board)
        score_t delta, eval_score
        piece_t captured_piece_type
    
    # Stand pat cutoffs
    if maximizing_player:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
    
    # Examine only captures for quiescence
    for move in board.legal_moves:
        if not board.is_capture(move):
            continue
            
        captured_piece = board.piece_at(move.to_square)
        
        # Delta pruning
        delta = piece_value[captured_piece.piece_type] if captured_piece else 100
        if maximizing_player and stand_pat + delta + 200 < alpha:
            continue
        elif not maximizing_player and stand_pat - delta - 200 > beta:
            continue
        
        board.push(move)
        eval_score = quiescence_cy(evaluate_static_func, board, alpha, beta, not maximizing_player, piece_value)
        board.pop()
        
        if maximizing_player:
            alpha = max(alpha, eval_score)
            if alpha >= beta:
                break
        else:
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
    
    return alpha if maximizing_player else beta

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int score_move_cy(board, move, int ply, tt_move, killer_moves, history, dict piece_value):
    """
    Optimized move scoring for move ordering.
    """
    cdef:
        int score = 0
        square_t from_square, to_square
        piece_t victim_type, attacker_type
        int victim_value, attacker_value
        object km

    # TT move gets highest priority
    if tt_move and move == tt_move:
        return 20000

    # Killer moves get high priority – ensure km is a list before subscripting
    km = killer_moves[ply]
    if isinstance(km, list):
        if (km[0] is not None and km[0] == move) or (km[1] is not None and km[1] == move):
            return 10000
    else:
        # Fallback if km is not a list: if equals move, act as a killer move.
        if km == move:
            return 10000

    # Score captures using MVV-LVA
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        attacker_piece = board.piece_at(move.from_square)

        if captured_piece:
            victim_value = piece_value[captured_piece.piece_type]
            attacker_value = piece_value[attacker_piece.piece_type]
            return 9000 + (victim_value * 10 - attacker_value)
        else:
            return 8000  # En passant

    # Promotions get high priority
    if move.promotion:
        return 9500 if move.promotion == chess.QUEEN else 8500

    # Checks get medium priority
    if board.gives_check(move):
        return 8000

    # Use history heuristic for quiet moves
    from_square = move.from_square
    to_square = move.to_square
    move_key = (from_square, to_square)

    return history.get(move_key, 0)
