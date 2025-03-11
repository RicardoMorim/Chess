import cProfile
from collections import OrderedDict, namedtuple
import copy
import json
import logging
import os
import pstats
from time import sleep
import chess
import chess.svg
import chess.engine
import chess.pgn
import stable
from multiprocessing import Pool, cpu_count


class ChessEncoder(json.JSONEncoder):
    """
    JSON encoder for chess-related objects.
    """

    KEY_SAN = "san"
    KEY_VALUE = "value"
    KEY_MOVE = "move"

    def default(self, obj):
        """
        Convert chess-related objects to JSON-compatible format.

        Args:
            obj: The object to be converted.

        Returns:
            The JSON-compatible representation of the object.
        """
        if type(obj) == chess.Move:
            return {self.KEY_SAN: obj.uci()}
        elif type(obj) == MoveEvaluation:
            return {
                self.KEY_VALUE: obj.value,
                self.KEY_MOVE: obj.move.uci() if obj.move else None,
            }
        return super().default(obj)


# Define a named tuple to represent a move and its evaluation
MoveEvaluation = namedtuple("MoveEvaluation", ["value", "move"])


class Minimax:
    def __init__(self, board, depth, color, oppenings):
        self.board = board
        self.max_depth = depth
        self.transposition_table = OrderedDict()
        self.color = color
        self.is_maximizing_player = True if self.color == chess.WHITE else False
        self.cache_file = "./cache/transposition_cache_Minimaxv2.json"
        self.computed_moves_cache_file = "./cache/computed_moves_cache_Minimaxv2.json"
        self.computed_positions = {}
        self.load_cache()
        self.openings = oppenings
        self.num_of_comp_pos = 0
        self.history_table = {}
        self.killer_moves = {}

    def store_transposition_table_entry(self, board, depth, best_move):
        """
        Store an entry in the transposition table.

        Parameters:
        - board: The chess board.
        - depth: The depth of the search.
        - best_move: The best move found.
        """
        key = self.calculate_board_hash(board)
        self.transposition_table[key] = {
            "depth": depth,
            "best_move": best_move,
            
        }

    def calculate_board_hash(self, board):
        """
        Calculate a hash value for the chess board.

        Parameters:
        - board: The chess board.

        Returns:
        - The calculated hash value.
        """
        return board.fen()

    def load_cache(self):
        """
        Load the transposition cache from the cache file.

        If the file doesn't exist or is empty, it initializes an empty cache.
        """
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        try:
            with open(self.cache_file, "r") as file:
                cache_data = json.load(file)
                self.transposition_table.update(cache_data)
                logging.info("Transposition cache loaded")
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.cache_file, "w") as file:
                json.dump({}, file)
                logging.info("Cache file didn't exist. Created a new cache file.")

        try:
            with open(self.computed_moves_cache_file, "r") as file:
                cache_data = json.load(file)
                self.computed_positions.update(cache_data)
                logging.info("computed moves cache loaded")
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.computed_moves_cache_file, "w") as file:
                json.dump({}, file)
                logging.info(
                    "computed moves cache file didn't exist. Created a new cache file."
                )

    def update_cache(self):
        """
        Update the transposition cache with the latest information.
        """
        transposition_copy = self.transposition_table.copy()
        with open(self.cache_file, "w") as file:
            json.dump(transposition_copy, file, cls=ChessEncoder)
        logging.info("done updating cache file")
        transposition_copy = copy.deepcopy(self.computed_positions)
        with open(self.computed_moves_cache_file, "w") as file:
            json.dump(transposition_copy, file, cls=ChessEncoder)

        logging.info("done updating cache file")

    def play_opening_move(self, board):
        """
        Play an opening move from the loaded openings.

        Parameters:
        - board: The current chess board.

        Returns:
        - The updated board after playing the move.
        - The move played.
        """
        for name, opening in self.openings.items():
            newBoard = chess.Board()
            for move in opening.mainline_moves():
                if newBoard == board:
                    sleep(1)
                    print(move, "-> from oppening: " + name)
                    return move
                newBoard.push(move)

        return self.minimax_root()

    def deepening_search_depth(self):
        """
        Increase the search depth as the game progresses.

        This function adjusts the maximum depth for the search algorithm based on the number of pieces on the board.
        """
        # Calculate the number of pieces on the board
        num_pieces = len(self.board.piece_map())

        # Adjust the maximum depth based on the number of pieces
        if num_pieces < 5:
            self.max_depth += 6
        elif num_pieces < 8:
            self.max_depth += 5
        elif num_pieces < 10:
            self.max_depth += 4
        elif num_pieces < 13:
            self.max_depth += 3
        elif num_pieces < 15:
            self.max_depth += 2
        elif num_pieces < 17:
            self.max_depth += 1
        print(self.max_depth)

    def getBestMove(self):

        if len(self.board.move_stack) < 20:
            return self.play_opening_move(self.board)

        self.deepening_search_depth()

        key = self.calculate_board_hash(self.board)
        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= self.max_depth:
                if entry["best_move"] is not None:
                    return chess.Move.from_uci(entry["best_move"]["san"])

        profiler = cProfile.Profile()
        profiler.enable()

        move = self.minimax_root()

        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        with open("./debug.txt", "w") as f:
            stats.stream = f
            stats.print_stats()

        self.store_transposition_table_entry(
            self.board,
            self.max_depth,
            move,
        )
        self.update_cache()

        print(self.num_of_comp_pos, "num of computed positions")

        return move

    def minimax_root(self):
        board = copy.deepcopy(self.board)

        # Get all legal moves and their heuristic scores
        newGameMoves = [(move, self.score_move(move, 1)) for move in board.legal_moves]
        newGameMoves.sort(key=lambda x: x[1], reverse=True)
        available_cpus = max(1, cpu_count() - 1)  # Leave one CPU free for other tasks

        # Set up multiprocessing pool with the number of CPUs available
        with Pool(processes=available_cpus) as pool:
            # Submit all tasks to the pool asynchronously
            async_results = [
                (
                    move,
                    pool.apply_async(
                        self.minimax,
                        args=(
                            self.max_depth - 1,
                            self.board_after_move(
                                board, move
                            ),  # Helper function to get board after move
                            -10000,
                            10000,
                            not self.is_maximizing_player,
                        ),
                    ),
                )
                for move, _ in newGameMoves
            ]

            # Wait for all processes to complete and gather results
            results = [
                (move, async_result.get()) for move, async_result in async_results
            ]

        # Find the best move from the completed results
        bestMoveFound = None
        bestScore = -float("inf") if self.is_maximizing_player else float("inf")

        for move, score in results:
            if self.is_maximizing_player:
                if score > bestScore:
                    bestScore = score
                    bestMoveFound = move
            else:
                if score < bestScore:
                    bestScore = score
                    bestMoveFound = move

        return bestMoveFound

    def board_after_move(self, board, move):
        board_copy = copy.deepcopy(board)
        board_copy.push(move)
        return board_copy

    def quiescence(self, board, alpha, beta, is_maximizing_player, depth=0, max_depth=0):
        if depth >= max_depth:
            return self.evaluateBoard(board)

        stand_pat = self.evaluateBoard(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        # Expand to tactical threats, not just captures
        newGameMoves = [move for move in board.legal_moves if board.is_capture(move) or board.gives_check(move) or self.detect_threats(board, move)]
        
        if not newGameMoves:
            return alpha

        for move in newGameMoves:
            board.push(move)
            score = -self.quiescence(board, -beta, -alpha, not is_maximizing_player, depth + 1)
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha
    
    def detect_threats(self, board, move):
        """
        Detect tactical threats like forks, pins, skewers, and discovered attacks
        for the given move.
        
        Parameters:
        - board: The current board state
        - move: The move to be checked for tactical threats
        
        Returns:
        - True if the move creates or avoids a tactical threat; otherwise, False
        """
        board.push(move)

        # Check for forks (multiple attacks after the move)
        if self.is_fork(board, move):
            board.pop()
            return True

        # Check for pins (pieces pinned to the king)
        if self.is_pin(board, move):
            board.pop()
            return True

        # Check for skewers (higher value piece behind a lower value piece)
        if self.is_skewer(board, move):
            board.pop()
            return True

        # Check for discovered attacks (moving a piece to expose an attack from another piece)
        if self.is_discovered_attack(board, move):
            board.pop()
            return True

        board.pop()
        return False

    def is_fork(self, board, move):
        """
        Check if a move results in a fork.
        
        A fork occurs when one piece attacks two or more enemy pieces simultaneously.
        
        Parameters:
        - board: The current board state
        - move: The move to check
        
        Returns:
        - True if the move results in a fork, otherwise False
        """
        attacker = board.piece_at(move.to_square)
        if not attacker:
            return False

        attacker_attacks = list(board.attacks(move.to_square))
        targets = [sq for sq in attacker_attacks if board.piece_at(sq) and board.piece_at(sq).color != attacker.color]
        
        # Fork if there are at least two enemy pieces attacked
        return len(targets) >= 2

    def is_pin(self, board, move):
        """
        Check if the move results in a pin.
        
        A pin occurs when a piece cannot move without exposing a more valuable piece (usually the king) to attack.
        
        Parameters:
        - board: The current board state
        - move: The move to check
        
        Returns:
        - True if the move results in a pin, otherwise False
        """
        king_square = board.king(board.turn)
        for direction in chess.RAYS:
            for square in chess.SquareSet(chess.ray_attack(move.to_square, king_square)):
                piece = board.piece_at(square)
                if piece and piece.color == board.turn and piece.piece_type != chess.KING:
                    if any(board.piece_at(chess.ray_attack(square, king_square))):
                        return True
        return False

    def is_skewer(self, board, move):
        """
        Check if the move results in a skewer.
        
        A skewer is like a pin, but the higher-value piece is in front of the lower-value piece on the same line of attack.
        
        Parameters:
        - board: The current board state
        - move: The move to check
        
        Returns:
        - True if the move results in a skewer, otherwise False
        """
        attacker = board.piece_at(move.to_square)
        if not attacker or attacker.piece_type == chess.KING:
            return False

        attack_squares = board.attacks(move.to_square)
        for square in attack_squares:
            piece = board.piece_at(square)
            if piece and piece.color != attacker.color:
                for behind_square in chess.SquareSet(chess.ray_attack(square, move.to_square)):
                    behind_piece = board.piece_at(behind_square)
                    if behind_piece and behind_piece.color != attacker.color:
                        if behind_piece.piece_type < piece.piece_type:  # Lower value piece behind
                            return True
        return False

    def is_discovered_attack(self, board, move):
        """
        Check if the move results in a discovered attack.
        
        A discovered attack occurs when a piece moves out of the way, exposing an attack from a hidden piece.
        
        Parameters:
        - board: The current board state
        - move: The move to check
        
        Returns:
        - True if the move results in a discovered attack, otherwise False
        """
        # First, get the piece that was moved
        piece_moved = board.piece_at(move.to_square)
        if not piece_moved:
            return False

        # Check the line of attack before and after the move to see if a hidden attack is exposed
        for direction in chess.RAYS:
            ray_attack_before = chess.SquareSet(chess.ray_attack(move.from_square, board.king(not board.turn)))
            ray_attack_after = chess.SquareSet(chess.ray_attack(move.to_square, board.king(not board.turn)))

            if ray_attack_after.difference(ray_attack_before):
                # Check if the revealed piece is attacking a valuable target
                hidden_attacker_square = ray_attack_after.pop()
                hidden_attacker = board.piece_at(hidden_attacker_square)
                if hidden_attacker and hidden_attacker.color != board.turn:
                    return True

        return False



    def minimax(self, depth, board, alpha, beta, is_maximizing_player):
        if board.is_game_over():
            if board.is_checkmate():
                return 100000 - depth if board.turn != self.color else -100000 + depth
            return 0
        if depth <= 0:
            return self.quiescence(board, alpha, beta, is_maximizing_player)

        # Check for immediate mate
        if self.is_immediate_mate(board):
            return 99999 - depth if is_maximizing_player else -99999 + depth

        newGameMoves = [(move, self.score_move(move, depth)) for move in board.legal_moves]
        newGameMoves.sort(key=lambda x: x[1], reverse=True)

        if is_maximizing_player:
            bestMove = -9999
            for move, _ in newGameMoves:
                board.push(move)
                bestMove = max(bestMove, self.minimax(depth - 1, board, alpha, beta, not is_maximizing_player))
                board.pop()
                alpha = max(alpha, bestMove)
                if beta <= alpha:
                    break
            return bestMove
        else:
            bestMove = 9999
            for move, _ in newGameMoves:
                board.push(move)
                bestMove = min(bestMove, self.minimax(depth - 1, board, alpha, beta, not is_maximizing_player))
                board.pop()
                beta = min(beta, bestMove)
                if beta <= alpha:
                    break
            return bestMove
    
    def is_immediate_mate(self, board):
        for move in board.legal_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return True
            board.pop()
        return False

    def score_move(self, move, depth):
        if self.board.is_capture(move):
            return self.mvv_lva(move)  # Prioritize captures based on value
        history_score = self.history_table.get(move, 0)
        killer_score = 2 if move == self.killer_moves.get(depth) else 0
        return history_score + killer_score

    def mvv_lva(self, move):
        victim_square = move.to_square
        attacker_square = move.from_square
        
        # Get the victim and attacker piece coordinates (file, rank)
        victim_file = chess.square_file(victim_square)
        victim_rank = chess.square_rank(victim_square)
        
        attacker_file = chess.square_file(attacker_square)
        attacker_rank = chess.square_rank(attacker_square)

        # Get piece values using correct coordinates
        victim_value = self.getPieceValue(self.board.piece_at(victim_square), victim_file, victim_rank)
        attacker_value = self.getPieceValue(self.board.piece_at(attacker_square), attacker_file, attacker_rank)

        # MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
        return victim_value - attacker_value  # Higher score for better captures




    def evaluate_piece_development(self, board):
        # Initialize score
        score = 0

        # Evaluate piece development for both sides
        for color in [chess.WHITE, chess.BLACK]:
            # Count the number of moves made by knights, bishops, rooks, and queen
            knight_moves = sum(
                1
                for move in board.move_stack
                if board.piece_at(move.from_square)
                and board.piece_at(move.from_square).color == color
                and board.piece_at(move.from_square).piece_type == chess.KNIGHT
            )
            bishop_moves = sum(
                1
                for move in board.move_stack
                if board.piece_at(move.from_square)
                and board.piece_at(move.from_square).color == color
                and board.piece_at(move.from_square).piece_type == chess.BISHOP
            )
            rook_moves = sum(
                1
                for move in board.move_stack
                if board.piece_at(move.from_square)
                and board.piece_at(move.from_square).color == color
                and board.piece_at(move.from_square).piece_type == chess.ROOK
            )
            queen_moves = sum(
                1
                for move in board.move_stack
                if board.piece_at(move.from_square)
                and board.piece_at(move.from_square).color == color
                and board.piece_at(move.from_square).piece_type == chess.QUEEN
            )

            # Give a bonus for each piece that is developed early in the game
            score += (
                0.1 * knight_moves
                + 0.1 * bishop_moves
                + 0.1 * rook_moves
                + 0.1 * queen_moves
            )

        return score

    def get_king_expusure(self, board):
        king_square = board.king(board.turn)
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        king_expusure = 0
        if king_file < 3 or king_file > 5:
            king_expusure += 1
        if king_rank == 0 or king_rank == 7:
            king_expusure += 1
        return king_expusure

    def eval_center_control(self, board):
        center_control = 0
        for i in range(3, 5):
            for j in range(3, 5):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    center_control += 1
        return center_control

    def evaluate_pawn_structure(self, board):
        # Initialize score
        score = 0
        # Evaluate pawn structure for both sides
        for color in [chess.WHITE, chess.BLACK]:
            pawns = board.pieces(chess.PAWN, color)
            # Evaluate isolated pawns
            isolated_pawns = self.evaluate_isolated_pawns(board, pawns)
            score += isolated_pawns
            # Evaluate doubled pawns
            doubled_pawns = self.evaluate_doubled_pawns(board, pawns)
            score += doubled_pawns
            # Evaluate backward pawns
            backward_pawns = self.evaluate_backward_pawns(board, pawns)
            score += backward_pawns
            # Evaluate passed pawns
            passed_pawns = self.evaluate_passed_pawns(board, pawns)
            score += passed_pawns
        return score

    def evaluate_isolated_pawns(self, board, pawns):
        # Initialize score
        score = 0
        for pawn in pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            # Check if the pawn has no neighboring pawns on adjacent files
            if not any(
                board.piece_at(chess.square(file + i, rank))
                for i in [-1, 1]
                if file < 7
            ):
                score -= 0.5
        return score

    def evaluate_doubled_pawns(self, board, pawns):
        # Initialize score
        score = 0
        for pawn in pawns:
            file = chess.square_file(pawn)
            # Count the number of pawns on the same file
            num_pawns = sum(1 for p in pawns if chess.square_file(p) == file)
            # Penalize for each additional pawn on the same file
            if num_pawns > 1:
                score -= 0.5 * (num_pawns - 1)
        return score

    def evaluate_backward_pawns(self, board, pawns):
        # Initialize score
        score = 0
        for pawn in pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            # Check if the pawn has no friendly pawns in front or on adjacent files
            if not any(
                board.piece_at(chess.square(file + i, rank + 1)) == chess.PAWN
                or board.piece_at(chess.square(file + i, rank)) == chess.PAWN
                for i in [-1, 0, 1]
                if file < 7 and rank < 7
            ):
                score -= 0.5
        return score

    def evaluate_passed_pawns(self, board, pawns):
        # Initialize score
        score = 0
        for pawn in pawns:
            file = chess.square_file(pawn)
            rank = chess.square_rank(pawn)
            # Check if the pawn has no enemy pawns in front or on adjacent files
            if not any(
                board.piece_at(chess.square(file + i, rank - 1)) == chess.PAWN
                or board.piece_at(chess.square(file + i, rank)) == chess.PAWN
                for i in [-1, 0, 1]
            ):
                score += 0.5
        return score

    def evaluate_tactics(self, board):
        # Initialize score
        score = 0

        # Detect tactical patterns such as forks, skewers, and pins
        for move in board.legal_moves:
            # Check for captures
            if board.is_capture(move):
                captured_piece = board.piece_at(move.to_square)
                if captured_piece:
                    # Check if the move results in a capture of a higher-value piece
                    if captured_piece.piece_type == chess.QUEEN:
                        score += 2
                    elif captured_piece.piece_type == chess.ROOK:
                        score += 1
                    elif (
                        captured_piece.piece_type == chess.BISHOP
                        or captured_piece.piece_type == chess.KNIGHT
                    ):
                        score += 0.5

            # Check for pawn promotions
            if move.promotion:
                score += 1

            # Check for checks
            if board.gives_check(move):
                score += 0.5

        return score

    def evaluateBoard(self, board):

        # # Generate a hash for the current board state
        fen = self.calculate_board_hash(board)

        # # If the board state has been evaluated before, return the stored value
        if fen in self.computed_positions:
            return self.computed_positions[fen]["eval"]

        if board.is_checkmate():
            if board.turn == chess.WHITE:
                eval = 9999
            else:
                eval = -9999

        if board.is_game_over():
            return 0
        eval = 0
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    # Get the basic value of the piece
                    piece_value = self.getPieceValue(piece, i, j)
                    eval += piece_value
        if len(board.move_stack) < 20:
            eval += self.evaluate_piece_development(board)
            eval += self.get_king_expusure(board)
            eval += self.eval_center_control(board)

        eval += self.evaluate_pawn_structure(board)
        eval += self.evaluate_tactics(board)

        self.computed_positions[fen] = {"eval": eval}
        self.num_of_comp_pos += 1

        return eval

    def getPieceValue(self, piece, x, y):

        def getAbsoluteValue(piece, isWhite, x, y):

            piece_type = piece.piece_type

            if piece_type == chess.PAWN:
                return 10 + (
                    stable.pawnEvalWhite[y][x]
                    if isWhite
                    else stable.pawnEvalBlack[y][x]
                )
            elif piece_type == chess.ROOK:
                return 50 + (
                    stable.rookEvalWhite[y][x]
                    if isWhite
                    else stable.rookEvalBlack[y][x]
                )
            elif piece_type == chess.KNIGHT:
                return 30 + stable.knightEval[y][x]
            elif piece_type == chess.BISHOP:
                return 30 + (
                    stable.bishopEvalWhite[y][x]
                    if isWhite
                    else stable.bishopEvalBlack[y][x]
                )
            elif piece_type == chess.QUEEN:
                return 90 + stable.evalQueen[y][x]
            elif piece_type == chess.KING:
                return 900 + (
                    stable.kingEvalWhite[y][x]
                    if isWhite
                    else stable.kingEvalBlack[y][x]
                )

        if piece == None:
            return 0
        absolute_value = getAbsoluteValue(piece, piece.color == "w", x, y)
        return absolute_value if piece.color == chess.WHITE else -absolute_value
