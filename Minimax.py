import chess
import chess.pgn
import chess.polyglot
import random
import numpy as np
from TranspositionTable import TranspositionTable

# Import Cython implementations if available
try:
    # import minimax_cy
    CYTHON_AVAILABLE =  False
    print("Cython acceleration enabled!")
except ImportError:
    CYTHON_AVAILABLE = False
    print("Cython module not found. Using Python implementation.")

class MinimaxAI:
    def __init__(self, openings, color, depth=4):
        """
        Initialize the Minimax AI with openings, color, and search depth.

        :param openings: Dictionary of opening names to chess.pgn.Game objects
        :param color: chess.WHITE or chess.BLACK, indicating AI's color
        :param depth: Search depth for Minimax algorithm
        """
        self.openings = openings
        self.color = color
        self.depth = depth
        self.opening_moves = self.process_openings()
        
        # Initialize transposition table
        self.tt = TranspositionTable()
        
        # History heuristic table for move ordering
        self.history = {}  # (from_square, to_square) -> score
        
        # Piece values in centipawns
        self.piece_value = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Simplified piece-square tables for white (in centipawns)
        self.pawn_pst_white = [
            0, 0, 0, 0, 0, 0, 0, 0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5, 5, 10, 25, 25, 10, 5, 5,
            0, 0, 0, 20, 20, 0, 0, 0,
            5, -5, -10, 0, 0, -10, -5, 5,
            5, 10, 10, -20, -20, 10, 10, 5,
            0, 0, 0, 0, 0, 0, 0, 0
        ]
        self.knight_pst_white = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20, 0, 0, 0, 0, -20, -40,
            -30, 0, 10, 15, 15, 10, 0, -30,
            -30, 5, 15, 20, 20, 15, 5, -30,
            -30, 0, 15, 20, 20, 15, 0, -30,
            -30, 5, 10, 15, 15, 10, 5, -30,
            -40, -20, 0, 5, 5, 0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]
        self.bishop_pst_white = self.knight_pst_white  # Simplified
        self.rook_pst_white = [
            0, 0, 0, 0, 0, 0, 0, 0,
            5, 10, 10, 10, 10, 10, 10, 5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            -5, 0, 0, 0, 0, 0, 0, -5,
            5, 10, 10, 10, 10, 10, 10, 5,
            0, 0, 0, 5, 5, 0, 0, 0
        ]
        self.queen_pst_white = [x / 2 for x in self.knight_pst_white]  # Scaled
        self.king_pst_white = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20, 20, 0, 0, 0, 0, 20, 20,
            20, 30, 10, 0, 0, 10, 30, 20
        ]
        
        self.pst_white = {
            chess.PAWN: self.pawn_pst_white,
            chess.KNIGHT: self.knight_pst_white,
            chess.BISHOP: self.bishop_pst_white,
            chess.ROOK: self.rook_pst_white,
            chess.QUEEN: self.queen_pst_white,
            chess.KING: self.king_pst_white
        }
        
        # Precompute piece-square tables for black side
        self.pst_black = {}
        for piece_type, table in self.pst_white.items():
            self.pst_black[piece_type] = [table[chess.square_mirror(i)] for i in range(64)]
        
        # For Cython, prepare flattened PST arrays
        if CYTHON_AVAILABLE:
            self._prepare_cython_tables()
        
        # Killer moves: store two killer moves per ply
        self.killer_moves = [[None, None] for _ in range(self.depth + 1)]

    def _prepare_cython_tables(self):
        """Prepare optimized tables for Cython functions"""
        # Create flattened numpy arrays for faster access in Cython
        piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]
        
        # Initialize arrays (square * 6 + piece_idx)
        self.pst_white_flat = np.zeros(64 * 6, dtype=np.int32)
        self.pst_black_flat = np.zeros(64 * 6, dtype=np.int32)
        
        # Populate the arrays
        for square in range(64):
            for i, piece_type in enumerate(piece_types):
                self.pst_white_flat[square * 6 + i] = self.pst_white[piece_type][square]
                self.pst_black_flat[square * 6 + i] = self.pst_black[piece_type][square]

    def process_openings(self):
        """
        Process opening games to map board positions (FEN) to recommended moves.

        :return: Dictionary mapping FEN strings to lists of possible moves
        """
        opening_moves = {}
        for opening_name, game in self.openings.items():
            board = chess.Board()
            node = game
            while node.variations:
                next_node = node.variation[0]
                move = next_node.move
                fen = board.fen()
                if fen not in opening_moves:
                    opening_moves[fen] = []
                opening_moves[fen].append(move)
                board.push(move)
                node = next_node
        return opening_moves

    def score_move(self, board, move, ply, tt_move=None):
        """
        Assign a score to a move for move ordering.

        :param board: Current chess board
        :param move: Move to score
        :param ply: Current ply in the search
        :param tt_move: Best move from transposition table, if available
        :return: Score for the move
        """
        if CYTHON_AVAILABLE:
            return minimax_cy.score_move_cy(board, move, ply, tt_move, 
                                           self.killer_moves, self.history, self.piece_value)
        
        # Original Python implementation
        if tt_move and move == tt_move:
            return 20000  # TT move highest priority

        if ply < len(self.killer_moves) and move in self.killer_moves[ply]:
            return 10000  # Prioritize killer moves

            
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            attacker_piece = board.piece_at(move.from_square)
            
            if captured_piece:
                # MVV-LVA: Most Valuable Victim - Least Valuable Attacker
                victim_value = self.piece_value[captured_piece.piece_type]
                attacker_value = self.piece_value[attacker_piece.piece_type]
                return 9000 + (victim_value * 10 - attacker_value)
            else:
                return 8000  # En passant
                
        if move.promotion:
            return 9500 if move.promotion == chess.QUEEN else 8500
            
        # Check if move gives check (more expensive calculation)
        if board.gives_check(move):
            return 8000
            
        # History heuristic
        move_key = (move.from_square, move.to_square)
        return self.history.get(move_key, 0)

    def evaluate_static(self, board):
        """
        Optimized static evaluation function.

        :param board: Current chess board
        :return: Score in centipawns, positive favors white
        """
        if CYTHON_AVAILABLE:
            return minimax_cy.evaluate_static_cy(self.piece_value, 
                                               self.pst_white_flat,
                                               self.pst_black_flat,
                                               board)
        
        # Original Python implementation
        material = 0
        pst_score = 0
        
        # Process all pieces in one loop to reduce overhead
        for square, piece in board.piece_map().items():
            piece_value = self.piece_value[piece.piece_type]
            if piece.color == chess.WHITE:
                material += piece_value
                pst_score += self.pst_white[piece.piece_type][square]
            else:
                material -= piece_value
                pst_score -= self.pst_black[piece.piece_type][square]
        
        king_safety = self.evaluate_king_safety(board)
        return material + pst_score + king_safety

    def evaluate_king_safety(self, board):
        """
        Evaluate king safety based on castling rights and pawn shield.

        :param board: Current chess board
        :return: Safety score, positive favors white
        """
        score = 0
        # Castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            score += 50
        if board.has_queenside_castling_rights(chess.WHITE):
            score += 50
        if board.has_kingside_castling_rights(chess.BLACK):
            score -= 50
        if board.has_queenside_castling_rights(chess.BLACK):
            score -= 50
        
        # Pawn shield for white king on g1
        if board.king(chess.WHITE) == chess.G1:
            if board.piece_at(chess.F2) == chess.Piece(chess.PAWN, chess.WHITE):
                score += 20
            if board.piece_at(chess.G2) == chess.Piece(chess.PAWN, chess.WHITE):
                score += 20
            if board.piece_at(chess.H2) == chess.Piece(chess.PAWN, chess.WHITE):
                score += 20
        
        # Pawn shield for black king on g8
        if board.king(chess.BLACK) == chess.G8:
            if board.piece_at(chess.F7) == chess.Piece(chess.PAWN, chess.BLACK):
                score -= 20
            if board.piece_at(chess.G7) == chess.Piece(chess.PAWN, chess.BLACK):
                score -= 20
            if board.piece_at(chess.H7) == chess.Piece(chess.PAWN, chess.BLACK):
                score -= 20
        
        return score

    def quiescence(self, board, alpha, beta, maximizing_player):
        """
        Quiescence search to evaluate only capture moves at search leaves.
        Includes delta pruning to avoid searching unpromising captures.
        
        :param board: Current chess board
        :param alpha: Alpha value for pruning
        :param beta: Beta value for pruning
        :param maximizing_player: True if maximizing, False if minimizing
        :return: Quiescent evaluation score
        """
        if CYTHON_AVAILABLE:
            return minimax_cy.quiescence_cy(self.evaluate_static, board, alpha, beta, 
                                          maximizing_player, self.piece_value)
        
        # Original Python implementation
        stand_pat = self.evaluate_static(board)
        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        for move in board.legal_moves:
            if not board.is_capture(move):
                continue
            captured_piece = board.piece_at(move.to_square)
            # Delta pruning - skip captures that can't improve position enough
            delta = self.piece_value[captured_piece.piece_type] if captured_piece else 100
            if maximizing_player and stand_pat + delta + 200 < alpha:  # 200 as margin
                continue
            elif not maximizing_player and stand_pat - delta - 200 > beta:
                continue
            
            board.push(move)
            eval = self.quiescence(board, alpha, beta, not maximizing_player)
            board.pop()
            
            if maximizing_player:
                alpha = max(alpha, eval)
                if alpha >= beta:
                    break
            else:
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        
        return alpha if maximizing_player else beta

    def alphabeta(self, board, depth, alpha, beta, maximizing_player, ply=0, is_pv_node=True):
        """
        Principal Variation Search (PVS) with Alpha-Beta pruning.

        :param board: Current chess board
        :param depth: Remaining search depth
        :param alpha: Alpha value for pruning
        :param beta: Beta value for pruning
        :param maximizing_player: True if maximizing, False if minimizing
        :param ply: Current ply in the search
        :param is_pv_node: True if this is a PV node (should search with full window)
        :return: Best evaluation score and best move
        """
        # Generate Zobrist hash for current position
        zobrist_hash = chess.polyglot.zobrist_hash(board)
        original_alpha = alpha

        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                return -99999 if board.turn == chess.WHITE else 99999, None
            elif board.is_stalemate() or board.is_insufficient_material():
                return 0, None

        # Probe transposition table
        tt_entry = self.tt.probe(zobrist_hash)
        if tt_entry and tt_entry[1] >= depth:
            flag, value, best_move = tt_entry[2], tt_entry[3], tt_entry[4]
            if flag == 'exact':
                return value, best_move
            elif flag == 'lower' and value > alpha:
                alpha = value
            elif flag == 'upper' and value < beta:
                beta = value
            if alpha >= beta:
                return value, best_move

        # Leaf node - use quiescence search
        if depth <= 0:
            return self.quiescence(board, alpha, beta, maximizing_player), None

        # Null Move Pruning
        if depth >= 2 and not board.is_check() and not is_pv_node and maximizing_player:
            # Avoid null move in endgame (simplified check)
            piece_count = len(board.piece_map())
            if piece_count > 6:  # Arbitrary threshold
                board.push(chess.Move.null())
                null_eval, _ = self.alphabeta(board, depth - 3, beta - 1, beta, False, ply + 1, False)
                board.pop()
                if null_eval >= beta:
                    return beta, None
        
        # Razoring - prune at low depths if static evaluation is far below alpha
        if depth >= 1 and depth <= 3 and not board.is_check() and not is_pv_node:
            # Increasing margin with depth
            razor_margin = 300 + (depth - 1) * 100
            static_eval = self.evaluate_static(board)
            
            # If static evaluation + margin is below alpha, it's unlikely this position will be good
            if maximizing_player and static_eval + razor_margin < alpha:
                # At depth 1, just return quiescence score
                if depth == 1:
                    q_eval = self.quiescence(board, alpha, beta, maximizing_player)
                    return q_eval, None
                
                # At depths 2-3, verify with reduced depth search
                razor_alpha = alpha - razor_margin
                razor_eval, _ = self.alphabeta(board, 1, razor_alpha, razor_alpha + 1, maximizing_player, ply, False)
                if razor_eval <= razor_alpha:
                    return razor_eval, None
            
            # Mirror logic for minimizing player
            elif not maximizing_player and static_eval - razor_margin > beta:
                if depth == 1:
                    q_eval = self.quiescence(board, alpha, beta, maximizing_player)
                    return q_eval, None
                
                razor_beta = beta + razor_margin
                razor_eval, _ = self.alphabeta(board, 1, razor_beta - 1, razor_beta, maximizing_player, ply, False)
                if razor_eval >= razor_beta:
                    return razor_eval, None

        # Futility Pruning
        if depth == 1 and not board.is_check() and not is_pv_node:
            static_eval = self.evaluate_static(board)
            futility_margin = 900  # Queen value
            if maximizing_player and static_eval + futility_margin <= alpha:
                return static_eval, None
            elif not maximizing_player and static_eval - futility_margin >= beta:
                return static_eval, None

        # Get legal moves and sort them
        moves = list(board.legal_moves)
        tt_move = tt_entry[4] if tt_entry else None
        moves.sort(key=lambda m: self.score_move(board, m, ply, tt_move), reverse=True)

        if not moves:
            return self.evaluate_static(board), None

        best_move = None
        best_eval = -99999 if maximizing_player else 99999
        search_pv = True  # Flag to track if we are searching the first move (PV)

        for i, move in enumerate(moves):
            board.push(move)
            
            # Apply Late Move Reduction (LMR)
            if i >= 4 and depth >= 3 and not board.is_check() and not board.is_capture(move) and not move.promotion:
                reduced_depth = depth - 2
            else:
                reduced_depth = depth - 1

            # Principal Variation Search
            if search_pv:
                # First move searched with full window
                eval, _ = self.alphabeta(board, reduced_depth, alpha, beta, not maximizing_player, ply + 1, is_pv_node)
                search_pv = False
            else:
                # Remaining moves searched with null window, re-search if promising
                eval, _ = self.alphabeta(board, reduced_depth, alpha, alpha + 1, not maximizing_player, ply + 1, False)
                
                # Re-search with full window if needed
                if alpha < eval < beta:
                    eval, _ = self.alphabeta(board, reduced_depth, alpha, beta, not maximizing_player, ply + 1, False)
            
            board.pop()

            if maximizing_player:
                if eval > best_eval:
                    best_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    # Update history for good quiet moves causing beta cutoff
                    if not board.is_capture(move):
                        move_key = (move.from_square, move.to_square)
                        self.history[move_key] = self.history.get(move_key, 0) + depth * depth
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    break
            else:
                if eval < best_eval:
                    best_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    # Update history for good quiet moves causing alpha cutoff
                    if not board.is_capture(move):
                        move_key = (move.from_square, move.to_square)
                        self.history[move_key] = self.history.get(move_key, 0) + depth * depth
                        self.killer_moves[ply][1] = self.killer_moves[ply][0]
                        self.killer_moves[ply][0] = move
                    break

        # Store in transposition table
        if best_move:
            flag = 'exact'
            if best_eval <= original_alpha:
                flag = 'upper'
            elif best_eval >= beta:
                flag = 'lower'
            self.tt.store(zobrist_hash, depth, flag, best_eval, best_move)

        return best_eval, best_move

    def get_best_move(self, board):
        """
        Return the best move for the current board position using iterative deepening
        and aspiration windows.

        :param board: Current chess board
        :return: Best move as a chess.Move object
        """
        # Use opening book if position is in opening database
        if board.fen() in self.opening_moves:
            return random.choice(self.opening_moves[board.fen()])
        
        best_move = None
        prev_eval = 0  # Initial evaluation guess
        window = 50    # Initial aspiration window width (in centipawns)
        
        # Iterative deepening loop
        for depth in range(1, self.depth + 1):
            # Use best_move from previous depth at the beginning of move ordering
            moves = list(board.legal_moves)
            if best_move in moves:
                moves.remove(best_move)
                moves.insert(0, best_move)
            
            # Set aspiration window based on previous depth evaluation
            alpha = prev_eval - window
            beta = prev_eval + window
            
            # Keep re-searching with wider windows if evaluation falls outside the window
            attempts = 0
            while attempts < 3:  # Limit re-searches to avoid infinite loops
                if self.color == chess.WHITE:
                    eval, new_best_move = self.alphabeta(board, depth, alpha, beta, True, 0)
                else:
                    eval, new_best_move = self.alphabeta(board, depth, alpha, beta, False, 0)
                
                # Check if evaluation is within window
                if self.color == chess.WHITE and eval <= alpha:  # Fail low
                    window = window * 2
                    alpha = eval - window
                    attempts += 1
                    continue
                elif self.color == chess.WHITE and eval >= beta:  # Fail high
                    window = window * 2
                    beta = eval + window
                    attempts += 1
                    continue
                elif self.color == chess.BLACK and eval >= beta:  # Fail high for black
                    window = window * 2
                    beta = eval + window
                    attempts += 1
                    continue
                elif self.color == chess.BLACK and eval <= alpha:  # Fail low for black
                    window = window * 2
                    alpha = eval - window
                    attempts += 1
                    continue
                
                # If we're here, the evaluation is within the window
                if new_best_move:
                    best_move = new_best_move
                prev_eval = eval
                break
            
            # If we reach max attempts, use whatever best_move we have
            if attempts >= 3 and new_best_move:
                best_move = new_best_move
        
        # In the unlikely case we have no best move, just pick the first legal move
        if not best_move and board.legal_moves:
            best_move = list(board.legal_moves)[0]
            prev_eval = self.evaluate_static(board)

        print(f"Best move found: {best_move} with evaluation: {prev_eval}")
        return best_move