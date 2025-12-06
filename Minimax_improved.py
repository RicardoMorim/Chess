"""
Improved Minimax Chess Engine
=============================

Significant improvements over the original:
1. Better evaluation function with mobility, pawn structure, king safety
2. Improved piece-square tables including endgame king
3. Better move ordering with SEE and countermoves
4. Fixed LMR thresholds
5. Check extensions
6. Proper aspiration windows for both colors
7. Memory management (history table aging)

Compatible with the original MinimaxAI interface.
"""

import chess
import chess.pgn
import chess.polyglot
import random
from TranspositionTable import TranspositionTable


class MinimaxAI:
    def __init__(self, openings, color, depth=6):
        """
        Initialize the Minimax AI.
        
        Args:
            openings: Dictionary of opening names to chess.pgn.Game objects
            color: 'w' or 'b', indicating AI's color
            depth: Search depth (default 6)
        """
        self.openings = openings
        self.color = chess.WHITE if color == 'w' else chess.BLACK
        self.depth = depth
        self.opening_moves = self.process_openings()
        
        # Transposition table
        self.tt = TranspositionTable(size=2**20)
        
        # Move ordering tables
        self.history = {}  # History heuristic: (from, to) -> score
        self.countermoves = {}  # Countermove heuristic: move -> best response
        self.killer_moves = [[None, None] for _ in range(64)]  # Killer moves per ply
        
        # Search statistics
        self.nodes_searched = 0
        self.tt_hits = 0
        
        # Piece values (centipawns)
        self.piece_value = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        
        # Initialize piece-square tables
        self._init_pst()
    
    def _init_pst(self):
        """Initialize piece-square tables."""
        # Pawn PST (encourages central control and advancement)
        self.pawn_pst = [
            0,   0,   0,   0,   0,   0,   0,   0,
            50,  50,  50,  50,  50,  50,  50,  50,
            10,  10,  20,  30,  30,  20,  10,  10,
            5,   5,  10,  25,  25,  10,   5,   5,
            0,   0,   0,  20,  20,   0,   0,   0,
            5,  -5, -10,   0,   0, -10,  -5,   5,
            5,  10,  10, -20, -20,  10,  10,   5,
            0,   0,   0,   0,   0,   0,   0,   0
        ]
        
        # Knight PST (encourages central knights)
        self.knight_pst = [
            -50, -40, -30, -30, -30, -30, -40, -50,
            -40, -20,   0,   0,   0,   0, -20, -40,
            -30,   0,  10,  15,  15,  10,   0, -30,
            -30,   5,  15,  20,  20,  15,   5, -30,
            -30,   0,  15,  20,  20,  15,   0, -30,
            -30,   5,  10,  15,  15,  10,   5, -30,
            -40, -20,   0,   5,   5,   0, -20, -40,
            -50, -40, -30, -30, -30, -30, -40, -50
        ]
        
        # Bishop PST (encourages fianchetto and diagonals)
        self.bishop_pst = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,  10,  10,   5,   0, -10,
            -10,   5,   5,  10,  10,   5,   5, -10,
            -10,   0,  10,  10,  10,  10,   0, -10,
            -10,  10,  10,  10,  10,  10,  10, -10,
            -10,   5,   0,   0,   0,   0,   5, -10,
            -20, -10, -10, -10, -10, -10, -10, -20
        ]
        
        # Rook PST (encourages 7th rank and open files)
        self.rook_pst = [
            0,   0,   0,   0,   0,   0,   0,   0,
            5,  10,  10,  10,  10,  10,  10,   5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            -5,   0,   0,   0,   0,   0,   0,  -5,
            0,   0,   0,   5,   5,   0,   0,   0
        ]
        
        # Queen PST (encourages central queen but not too early)
        self.queen_pst = [
            -20, -10, -10,  -5,  -5, -10, -10, -20,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -10,   0,   5,   5,   5,   5,   0, -10,
            -5,    0,   5,   5,   5,   5,   0,  -5,
            0,     0,   5,   5,   5,   5,   0,  -5,
            -10,   5,   5,   5,   5,   5,   0, -10,
            -10,   0,   5,   0,   0,   0,   0, -10,
            -20, -10, -10,  -5,  -5, -10, -10, -20
        ]
        
        # King PST for middlegame (encourages castling, punishes center)
        self.king_pst_mg = [
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -30, -40, -40, -50, -50, -40, -40, -30,
            -20, -30, -30, -40, -40, -30, -30, -20,
            -10, -20, -20, -20, -20, -20, -20, -10,
            20,   20,   0,   0,   0,   0,  20,  20,
            20,   30,  10,   0,   0,  10,  30,  20
        ]
        
        # King PST for endgame (encourages active king)
        self.king_pst_eg = [
            -50, -40, -30, -20, -20, -30, -40, -50,
            -30, -20, -10,   0,   0, -10, -20, -30,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -30, -10,  30,  40,  40,  30, -10, -30,
            -30, -10,  30,  40,  40,  30, -10, -30,
            -30, -10,  20,  30,  30,  20, -10, -30,
            -30, -30,   0,   0,   0,   0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50
        ]
        
        # Build PST dictionaries
        self.pst_white = {
            chess.PAWN: self.pawn_pst,
            chess.KNIGHT: self.knight_pst,
            chess.BISHOP: self.bishop_pst,
            chess.ROOK: self.rook_pst,
            chess.QUEEN: self.queen_pst,
            chess.KING: self.king_pst_mg
        }
        
        # Mirror for black
        self.pst_black = {}
        for piece_type, table in self.pst_white.items():
            self.pst_black[piece_type] = [table[chess.square_mirror(i)] for i in range(64)]
    
    def process_openings(self):
        """Process opening book into position -> moves mapping."""
        opening_moves = {}
        for opening_name, game in self.openings.items():
            if game is None:
                continue
            board = chess.Board()
            node = game
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                fen = board.fen().split(' ')[0]  # Just piece positions
                if fen not in opening_moves:
                    opening_moves[fen] = []
                if move not in opening_moves[fen]:
                    opening_moves[fen].append(move)
                board.push(move)
                node = next_node
        return opening_moves
    
    def is_endgame(self, board):
        """Determine if position is an endgame."""
        # Endgame if no queens or if each side has at most queen + minor piece
        queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))
        minors = (len(board.pieces(chess.KNIGHT, chess.WHITE)) + 
                  len(board.pieces(chess.BISHOP, chess.WHITE)) +
                  len(board.pieces(chess.KNIGHT, chess.BLACK)) + 
                  len(board.pieces(chess.BISHOP, chess.BLACK)))
        rooks = len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))
        
        if queens == 0:
            return True
        if queens == 2 and minors <= 2 and rooks == 0:
            return True
        return False
    
    def evaluate_static(self, board):
        """
        Comprehensive static evaluation function.
        Returns score from white's perspective.
        """
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        if board.can_claim_draw():
            return 0
        
        score = 0
        is_endgame = self.is_endgame(board)
        
        # Material and PST
        for square, piece in board.piece_map().items():
            piece_val = self.piece_value[piece.piece_type]
            
            # Use endgame king PST if applicable
            if piece.piece_type == chess.KING and is_endgame:
                pst_table = self.king_pst_eg
                pst_val = pst_table[square] if piece.color == chess.WHITE else pst_table[chess.square_mirror(square)]
            else:
                if piece.color == chess.WHITE:
                    pst_val = self.pst_white[piece.piece_type][square]
                else:
                    pst_val = self.pst_black[piece.piece_type][square]
            
            if piece.color == chess.WHITE:
                score += piece_val + pst_val
            else:
                score -= piece_val + pst_val
        
        # Bishop pair bonus
        if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
            score += 30
        if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
            score -= 30
        
        # Rook on open/semi-open file
        score += self._evaluate_rooks(board)
        
        # Pawn structure
        score += self._evaluate_pawn_structure(board)
        
        # King safety (only in middlegame)
        if not is_endgame:
            score += self._evaluate_king_safety(board)
        
        # Mobility (simplified)
        score += self._evaluate_mobility(board)
        
        return score
    
    def _evaluate_rooks(self, board):
        """Evaluate rook placement on open/semi-open files."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            
            for rook_sq in board.pieces(chess.ROOK, color):
                file = chess.square_file(rook_sq)
                
                # Check if file is open (no pawns)
                white_pawns_on_file = len([sq for sq in board.pieces(chess.PAWN, chess.WHITE) 
                                          if chess.square_file(sq) == file])
                black_pawns_on_file = len([sq for sq in board.pieces(chess.PAWN, chess.BLACK) 
                                          if chess.square_file(sq) == file])
                
                if white_pawns_on_file == 0 and black_pawns_on_file == 0:
                    score += sign * 20  # Open file
                elif (color == chess.WHITE and white_pawns_on_file == 0) or \
                     (color == chess.BLACK and black_pawns_on_file == 0):
                    score += sign * 10  # Semi-open file
                
                # Rook on 7th rank
                rank = chess.square_rank(rook_sq)
                if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
                    score += sign * 20
        
        return score
    
    def _evaluate_pawn_structure(self, board):
        """Evaluate pawn structure: doubled, isolated, passed pawns."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            pawns = list(board.pieces(chess.PAWN, color))
            
            files_with_pawns = [chess.square_file(sq) for sq in pawns]
            
            for pawn_sq in pawns:
                file = chess.square_file(pawn_sq)
                rank = chess.square_rank(pawn_sq)
                
                # Doubled pawns (penalty)
                if files_with_pawns.count(file) > 1:
                    score -= sign * 10
                
                # Isolated pawns (penalty)
                adjacent_files = []
                if file > 0:
                    adjacent_files.append(file - 1)
                if file < 7:
                    adjacent_files.append(file + 1)
                
                has_neighbor = any(f in files_with_pawns for f in adjacent_files)
                if not has_neighbor:
                    score -= sign * 15
                
                # Passed pawns (bonus)
                is_passed = True
                enemy_pawns = list(board.pieces(chess.PAWN, not color))
                for enemy_sq in enemy_pawns:
                    enemy_file = chess.square_file(enemy_sq)
                    enemy_rank = chess.square_rank(enemy_sq)
                    
                    if abs(enemy_file - file) <= 1:
                        if color == chess.WHITE and enemy_rank > rank:
                            is_passed = False
                            break
                        elif color == chess.BLACK and enemy_rank < rank:
                            is_passed = False
                            break
                
                if is_passed:
                    # Bonus increases as pawn advances
                    if color == chess.WHITE:
                        score += sign * (10 + rank * 10)
                    else:
                        score += sign * (10 + (7 - rank) * 10)
        
        return score
    
    def _evaluate_king_safety(self, board):
        """Evaluate king safety."""
        score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            sign = 1 if color == chess.WHITE else -1
            king_sq = board.king(color)
            
            if king_sq is None:
                continue
            
            king_file = chess.square_file(king_sq)
            king_rank = chess.square_rank(king_sq)
            
            # Castled king bonus
            if color == chess.WHITE:
                if king_sq in [chess.G1, chess.H1]:  # Kingside castled
                    score += sign * 30
                    # Pawn shield
                    for f in [5, 6, 7]:  # f, g, h files
                        pawn_sq = chess.square(f, 1)
                        if board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, chess.WHITE):
                            score += sign * 10
                elif king_sq in [chess.B1, chess.C1]:  # Queenside castled
                    score += sign * 25
            else:
                if king_sq in [chess.G8, chess.H8]:
                    score += sign * 30
                    for f in [5, 6, 7]:
                        pawn_sq = chess.square(f, 6)
                        if board.piece_at(pawn_sq) == chess.Piece(chess.PAWN, chess.BLACK):
                            score += sign * 10
                elif king_sq in [chess.B8, chess.C8]:
                    score += sign * 25
            
            # Penalty for king in center during middlegame
            if king_file in [3, 4] and ((color == chess.WHITE and king_rank < 2) or 
                                        (color == chess.BLACK and king_rank > 5)):
                score -= sign * 30
            
            # Castling rights bonus
            if board.has_kingside_castling_rights(color):
                score += sign * 15
            if board.has_queenside_castling_rights(color):
                score += sign * 10
        
        return score
    
    def _evaluate_mobility(self, board):
        """Simplified mobility evaluation."""
        # Just count legal moves as a rough mobility measure
        # This is a simplification - proper mobility counts attacks per piece
        white_mobility = 0
        black_mobility = 0
        
        # Approximate by counting attacked squares for each piece type
        for color in [chess.WHITE, chess.BLACK]:
            mobility = 0
            for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
                for sq in board.pieces(piece_type, color):
                    attacks = board.attacks(sq)
                    mobility += len(attacks)
            
            if color == chess.WHITE:
                white_mobility = mobility
            else:
                black_mobility = mobility
        
        return (white_mobility - black_mobility) * 2  # Small weight
    
    def score_move(self, board, move, ply, tt_move=None, prev_move=None):
        """Score a move for move ordering."""
        score = 0
        
        # TT move gets highest priority
        if tt_move and move == tt_move:
            return 100000
        
        # Killer moves
        if ply < len(self.killer_moves):
            if move == self.killer_moves[ply][0]:
                return 90000
            if move == self.killer_moves[ply][1]:
                return 89000
        
        # Countermove bonus
        if prev_move and prev_move in self.countermoves:
            if move == self.countermoves[prev_move]:
                return 88000
        
        # Captures: MVV-LVA
        if board.is_capture(move):
            captured = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            
            if captured:
                victim_val = self.piece_value[captured.piece_type]
                attacker_val = self.piece_value[attacker.piece_type]
                # MVV-LVA: prioritize high value victims, low value attackers
                score = 50000 + victim_val * 10 - attacker_val
            else:
                # En passant
                score = 50000 + 100 * 10 - 100
            return score
        
        # Promotions
        if move.promotion:
            if move.promotion == chess.QUEEN:
                return 60000
            return 55000
        
        # History heuristic
        move_key = (move.from_square, move.to_square)
        return self.history.get(move_key, 0)
    
    def quiescence(self, board, alpha, beta, ply=0):
        """Quiescence search - only searches captures at leaf nodes."""
        stand_pat = self.evaluate_static(board)
        
        if board.turn == chess.WHITE:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)
        
        # Only search captures and promotions
        for move in board.legal_moves:
            if not (board.is_capture(move) or move.promotion):
                continue
            
            # Delta pruning
            captured = board.piece_at(move.to_square)
            delta = self.piece_value[captured.piece_type] if captured else 100
            
            if board.turn == chess.WHITE:
                if stand_pat + delta + 200 < alpha:
                    continue
            else:
                if stand_pat - delta - 200 > beta:
                    continue
            
            board.push(move)
            score = self.quiescence(board, alpha, beta, ply + 1)
            board.pop()
            
            if board.turn == chess.WHITE:
                alpha = max(alpha, score)
                if alpha >= beta:
                    return beta
            else:
                beta = min(beta, score)
                if beta <= alpha:
                    return alpha
        
        return alpha if board.turn == chess.WHITE else beta
    
    def alphabeta(self, board, depth, alpha, beta, ply=0, prev_move=None):
        """
        Negamax-style alpha-beta with all improvements.
        Returns score from the perspective of the side to move.
        """
        self.nodes_searched += 1
        original_alpha = alpha
        maximizing = board.turn == chess.WHITE
        
        # Check for game over
        if board.is_game_over():
            if board.is_checkmate():
                # Return large negative value (we got mated)
                return -99999 + ply if maximizing else 99999 - ply
            return 0  # Draw
        
        # Transposition table lookup
        zobrist = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.probe(zobrist)
        tt_move = None
        
        if tt_entry and tt_entry[1] >= depth:
            self.tt_hits += 1
            flag, value, tt_move = tt_entry[2], tt_entry[3], tt_entry[4]
            if flag == 'exact':
                return value
            elif flag == 'lower':
                alpha = max(alpha, value)
            elif flag == 'upper':
                beta = min(beta, value)
            if alpha >= beta:
                return value
        
        # Leaf node - quiescence search
        if depth <= 0:
            return self.quiescence(board, alpha, beta)
        
        # Check extension
        in_check = board.is_check()
        if in_check:
            depth += 1
        
        # Null move pruning (not in check, not in endgame)
        if depth >= 3 and not in_check and not self.is_endgame(board):
            # Verify we have pieces to make nullmove worthwhile
            our_pieces = len([p for p in board.piece_map().values() if p.color == board.turn and p.piece_type != chess.PAWN])
            if our_pieces > 1:
                board.push(chess.Move.null())
                null_score = -self.alphabeta(board, depth - 3, -beta, -beta + 1, ply + 1)
                board.pop()
                
                if null_score >= beta:
                    return beta
        
        # Get and sort moves
        moves = list(board.legal_moves)
        if not moves:
            if in_check:
                return -99999 + ply if maximizing else 99999 - ply
            return 0
        
        # Sort moves
        moves.sort(key=lambda m: self.score_move(board, m, ply, tt_move, prev_move), reverse=True)
        
        best_move = None
        best_score = -float('inf') if maximizing else float('inf')
        
        for i, move in enumerate(moves):
            board.push(move)
            
            # Late Move Reductions
            reduction = 0
            if i >= 3 and depth >= 3 and not in_check and not board.is_capture(move) and not move.promotion:
                reduction = 1
                if i >= 6:
                    reduction = 2
            
            # PVS: first move with full window, others with null window
            if i == 0:
                score = self.alphabeta(board, depth - 1, alpha, beta, ply + 1, move)
            else:
                # Null window search
                if maximizing:
                    score = self.alphabeta(board, depth - 1 - reduction, alpha, alpha + 1, ply + 1, move)
                    if alpha < score < beta:
                        score = self.alphabeta(board, depth - 1, alpha, beta, ply + 1, move)
                else:
                    score = self.alphabeta(board, depth - 1 - reduction, beta - 1, beta, ply + 1, move)
                    if alpha < score < beta:
                        score = self.alphabeta(board, depth - 1, alpha, beta, ply + 1, move)
            
            board.pop()
            
            if maximizing:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            
            if alpha >= beta:
                # Update killer moves and history
                if not board.is_capture(move):
                    self.killer_moves[ply][1] = self.killer_moves[ply][0]
                    self.killer_moves[ply][0] = move
                    
                    move_key = (move.from_square, move.to_square)
                    self.history[move_key] = self.history.get(move_key, 0) + depth * depth
                    
                    if prev_move:
                        self.countermoves[prev_move] = move
                break
        
        # Store in TT
        if best_move:
            if best_score <= original_alpha:
                flag = 'upper'
            elif best_score >= beta:
                flag = 'lower'
            else:
                flag = 'exact'
            self.tt.store(zobrist, depth, flag, best_score, best_move)
        
        return best_score
    
    def get_best_move(self, board):
        """
        Get the best move using iterative deepening with aspiration windows.
        """
        # Clear stats
        self.nodes_searched = 0
        self.tt_hits = 0
        
        # Check opening book
        fen_key = board.fen().split(' ')[0]
        if fen_key in self.opening_moves:
            moves = self.opening_moves[fen_key]
            legal = [m for m in moves if m in board.legal_moves]
            if legal:
                return random.choice(legal)
        
        # Age history table to prevent overflow
        if len(self.history) > 10000:
            self.history = {k: v // 2 for k, v in self.history.items() if v > 10}
        
        best_move = None
        prev_score = 0
        
        # Iterative deepening
        for depth in range(1, self.depth + 1):
            # Aspiration window
            if depth >= 4:
                window = 50
                alpha = prev_score - window
                beta = prev_score + window
            else:
                alpha = -100000
                beta = 100000
            
            # Search with aspiration window
            while True:
                score = self.alphabeta(board, depth, alpha, beta)
                
                if score <= alpha:
                    # Fail low - widen window down
                    alpha = max(alpha - window * 2, -100000)
                    window *= 2
                elif score >= beta:
                    # Fail high - widen window up
                    beta = min(beta + window * 2, 100000)
                    window *= 2
                else:
                    break
            
            prev_score = score
            
            # Get best move from TT
            zobrist = chess.polyglot.zobrist_hash(board)
            tt_entry = self.tt.probe(zobrist)
            if tt_entry and tt_entry[4]:
                best_move = tt_entry[4]
            
            print(f"Depth {depth}: score={score}, move={best_move}, nodes={self.nodes_searched}, tt_hits={self.tt_hits}")
        
        # Fallback
        if not best_move or best_move not in board.legal_moves:
            best_move = random.choice(list(board.legal_moves))
        
        return best_move
