from collections import OrderedDict, namedtuple
import copy
import hashlib
import json
import logging
import os
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
        self.is_maximizing_player = True if self.color == "black" else False
        self.cache_file = "./cache/transposition_cache_Minimaxv2.json"
        self.computed_moves_cache_file = "./cache/computed_moves_cache_Minimaxv2.json"
        self.computed_positions = {}
        self.load_cache()
        self.openings = oppenings
        self.num_of_comp_pos = 0

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
            print(transposition_copy, "computed positions")

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
        for _, opening in self.openings.items():
            newBoard = chess.Board()
            for move in opening.mainline_moves():
                if newBoard == board:
                    sleep(1)
                    print(move)
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
                return chess.Move.from_uci(entry["best_move"]["san"])

        move = self.minimax_root()

        self.store_transposition_table_entry(
            self.board,
            self.max_depth,
            move,
        )

        print("Computed positions:", self.computed_positions)

        self.update_cache()

        print(self.num_of_comp_pos, "num of computed positions")

        return move

    def minimax_root(self):
        board = copy.deepcopy(self.board)
        newGameMoves = [
            (move, self.score_move(move, board)) for move in board.legal_moves
        ]

        newGameMoves.sort(key=lambda x: x[1], reverse=True)

        # Create a multiprocessing pool
        pool = Pool(processes=cpu_count())

        # Create a list to hold the results
        results = []

        for move, _ in newGameMoves:
            # Create a new copy of the board for each move
            board.push(move)
            if board.is_checkmate():
                return move
            # Start a new process for each move
            result = pool.apply_async(
                self.minimax,
                args=(
                    self.max_depth - 1,
                    board,
                    -10000,
                    10000,
                    self.is_maximizing_player,
                ),
            )
            results.append((move, result))
            board.pop()

        # Close the pool and wait for all processes to finish
        pool.close()
        pool.join()

        # Get the results from each process
        results = [(move, result.get()) for move, result in results]

        # Find the move with the highest value
        bestMoveFound = max(results, key=lambda x: x[1])[0]

        return bestMoveFound

    def quiescence(
        self, board, alpha, beta, is_maximizing_player, depth=0, max_depth=5
    ):
        if depth >= max_depth:
            return self.evaluateBoard(board)

        stand_pat = self.evaluateBoard(board)
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat

        newGameMoves = [
            move
            for move in board.legal_moves
            if board.is_capture(move) or board.gives_check(move)
        ]
        if not newGameMoves:
            return alpha

        for move in newGameMoves:
            board.push(move)
            score = -self.quiescence(
                board, -beta, -alpha, not is_maximizing_player, depth + 1
            )
            board.pop()

            if score >= beta:
                return beta
            if score > alpha:
                alpha = score

        return alpha

    def minimax(self, depth, board, alpha, beta, is_maximizing_player):
        if board.is_game_over():
            return self.evaluateBoard(board)

        if depth <= 0:
            return -self.quiescence(board, beta, alpha, is_maximizing_player)

        # Null move pruning
        if depth >= 3 and not board.is_check():
            board.push(chess.Move.null())
            score = -self.minimax(
                depth - 3, board, -beta, -beta + 1, not is_maximizing_player
            )
            board.pop()
            if score >= beta:
                return beta

        newGameMoves = [
            (move, self.score_move(move, board))
            for i, (move) in enumerate(board.legal_moves)
        ]
        newGameMoves.sort(key=lambda x: x[1], reverse=True)
        n_of_legal_moves = len(list(board.legal_moves))

        bestMove = -9999 if is_maximizing_player else 9999
        for i, (move, score) in enumerate(newGameMoves):
            board.push(move)
            new_depth = depth - 1
            if i > n_of_legal_moves / 2:
                new_depth = depth - 2  # Late Move Reductions

            if i == 0 or score > alpha:  # Principal Variation Search
                score = -self.minimax(
                    new_depth, board, -beta, -alpha, not is_maximizing_player
                )
            else:
                score = -self.minimax(
                    new_depth, board, -alpha - 1, -alpha, not is_maximizing_player
                )
                if alpha < score and score < beta:
                    score = -self.minimax(
                        new_depth, board, -beta, -alpha, not is_maximizing_player
                    )
            board.pop()

            if is_maximizing_player:
                if score > bestMove:
                    bestMove = score
                alpha = max(alpha, bestMove)
            else:
                if score < bestMove:
                    bestMove = score
                beta = min(beta, bestMove)

            if beta <= alpha:
                break

        return bestMove

    def score_move(self, move, board):
        # Initialize score
        score = 0

        # Evaluate captures
        if board.is_capture(move):
            score += 100

        # Evaluate checks
        if board.is_check():
            score += 30

        # Evaluate castles
        if board.is_castling(move):
            score += 20

        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    # Get the basic value of the piece
                    piece_value = self.getPieceValue(piece, i, j)
                    score += piece_value

        score *= 1 if board.turn == chess.WHITE else -1

        return score

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
        # for i in range(8):
        #     for j in range(8):
        #         square = chess.square(j, 7 - i)
        #         piece = board.piece_at(square)
        #         if piece is not None:
        #             # Get the basic value of the piece
        #             piece_value = self.getPieceValue(piece, i, j)
        #             eval += piece_value

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
