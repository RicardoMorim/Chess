import chess as ch
import os
from collections import OrderedDict, namedtuple
import json
import logging
import chess.engine


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
        if type(obj) == ch.Move:
            return {self.KEY_SAN: obj.uci()}
        elif type(obj) == MoveEvaluation:
            return {
                self.KEY_VALUE: obj.value,
                self.KEY_MOVE: obj.move.uci() if obj.move else None,
            }
        return super().default(obj)


# Define a named tuple to represent a move and its evaluation
MoveEvaluation = namedtuple("MoveEvaluation", ["value", "move"])

# Piece values for evaluation
piece_values = {
    ch.PAWN: 1,
    ch.ROOK: 5,
    ch.BISHOP: 3,
    ch.KNIGHT: 3,
    ch.QUEEN: 8,
    ch.KING: 999,
}


class Engine:
    def __init__(
        self,
        board,
        maxDepth,
        color,
        # cache_file="E:/chess_cache/transposition_cache.json",
        cache_file="./cache/transposition_cache_Minimax.json",
        stockfish_path="./stockfish/stockfish-windows-x86-64-avx2.exe",
    ):
        """
        Initialize the chess engine.

        Parameters:
        - board: The chess board.
        - maxDepth: The maximum depth for the search algorithm.
        - color: The color of the engine.
        - cache_file: The file to store the transposition cache.

        This function loads the transposition cache from the cache file.
        """
        self.board = board
        self.color = color
        self.maxDepth = maxDepth
        self.transposition_table = OrderedDict()
        self.cache_file = cache_file
        self.load_cache()
        self.engine_eval = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.engine_eval.configure({"Threads": 8})

    def store_transposition_table_entry(self, board, depth, value, best_move, flag):
        """
        Store an entry in the transposition table.

        Parameters:
        - board: The chess board.
        - depth: The depth of the search.
        - value: The evaluation value.
        - best_move: The best move found.
        - flag: The flag indicating the type of result (exact, lowerbound, upperbound).
        """
        key = self.calculate_board_hash(board)
        self.transposition_table[key] = {
            "depth": depth,
            "value": value,
            "best_move": best_move,
            "flag": flag,
        }

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

    def update_cache(self):
        """
        Update the transposition cache with the latest information.

        Parameters:
        - board: The chess board.
        - depth: The depth of the search.
        - value: The evaluation value.
        - best_move: The best move found.
        - flag: The flag indicating the type of result (exact, lowerbound, upperbound).
        """
        transposition_copy = self.transposition_table.copy()
        with open(self.cache_file, "w") as file:
            json.dump(transposition_copy, file, cls=ChessEncoder)
        logging.info("done updating cache file")

    def getBestMove(self):
        """
        Get the best move using the engine.

        Returns:
        - The best move found by the engine.
        """

        return self.engine()

    def stockfish_evalFunct(self, board):
        """
        Evaluate the current position based on material and positional factors.

        Returns:
        - The evaluation score for the current position.
        """
        try:
            result = self.engine_eval.analyse(board, chess.engine.Limit(time=0.1))
            score = result["score"].relative.score()
            return score
        except ValueError:
            logging.error("Error parsing Stockfish output.")
            logging.error("Using built-in eval...")
            return self.built_in_evalFunc(board)

    def built_in_evalFunc(self, board):
        compt = 0
        # Sums up the material values
        for square in board.piece_map():
            compt += self.squareResPoints(square, board)
        compt *= 0.7
        compt += self.mateOpportunity(board) * 0.15 + self.openning(board) * 0.15
        return compt

    def mateOpportunity(self, board):
        """
        Check if the current position presents a checkmate opportunity.

        Returns:
        - A large positive score if the current player has a checkmate opportunity.
        - A large negative score if the opponent has a checkmate opportunity.
        - Otherwise, returns 0.
        """
        if board.is_checkmate():
            if board.turn == self.color:
                return 999999
            else:
                return -999999
        else:
            return 0

    def openning(self, board):
        """
        Evaluate the opening phase of the game.

        Returns:
        - A score based on the number of legal moves, adjusted for the opening phase.
        """
        legal_moves = list(board.legal_moves)
        if board.fullmove_number < 10:
            if board.turn == self.color:
                return 1 / 30 * len(legal_moves)
            else:
                return -1 / 30 * len(legal_moves)
        else:
            opening_evaluation = self.calculate_opening_evaluation(board)
            return opening_evaluation

    def calculate_opening_evaluation(self, board):
        """
        Calculate the overall opening evaluation based on material, pawn structure, and mobility.

        Returns:
        - The opening evaluation score.
        """
        return (
            0.5 * self.calculate_material_evaluation(board)
            + 0.3 * self.calculate_pawn_structure_evaluation(board)
            + 0.2 * self.calculate_mobility_evaluation(board)
        )

    def calculate_material_evaluation(self, board):
        """
        Calculate the material evaluation based on the piece values.

        Returns:
        - The material evaluation score.
        """
        material_eval = 0
        for square, piece in board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == self.color:
                material_eval += value
            else:
                material_eval -= value
        return material_eval

    def calculate_pawn_structure_evaluation(self, board):
        """
        Calculate the pawn structure evaluation.

        Returns:
        - The pawn structure evaluation score.
        """
        pawn_structure_eval = 0
        for square, piece in board.piece_map().items():
            if piece.piece_type == ch.PAWN and piece.color == self.color:
                file, rank = ch.square_file(square), ch.square_rank(square)
                pawn_structure_eval += 0.1 * (4 - abs(3 - file))
        return pawn_structure_eval

    def calculate_mobility_evaluation(self, board):
        """
        Calculate the mobility evaluation based on legal moves.

        Returns:
        - The mobility evaluation score.
        """
        mobility_eval = 0
        for move in board.legal_moves:
            if board.turn == self.color:
                mobility_eval += 1  # Increase for own legal moves
            else:
                mobility_eval -= 1  # Decrease for opponent's legal moves
        return mobility_eval

    def squareResPoints(self, square, board):
        """
        Calculate the evaluation points for a given square.

        Parameters:
        - square: The chess square.

        Returns:
        - The evaluation points for the specified square.
        """
        piece_type = board.piece_type_at(square)
        piece_value = piece_values.get(piece_type, 0)

        if board.color_at(square) != board.turn:
            return -piece_value
        else:
            if piece_type == ch.PAWN:
                file, rank = ch.square_file(square), ch.square_rank(square)
                pawn_structure_value = 0.1 * (4 - abs(3 - file))
                return piece_value + pawn_structure_value
            else:
                return piece_value

    def engine(self):
        """
        Perform the main engine search.

        Returns:
        - The best move found by the engine.
        """
        move, _ = self.minimax(self.board.copy(), float("-inf"), float("inf"), 1)
        self.update_cache()
        return move

    def calculate_complexity(self):
        """
        Calculate the complexity of the position.

        Returns:
        - The calculated complexity score.
        """
        material_complexity = abs(self.calculate_material_evaluation())
        pawn_structure_complexity = self.calculate_pawn_structure_evaluation()
        king_safety_complexity = self.calculate_king_safety_evaluation()

        total_complexity = (
            material_complexity + pawn_structure_complexity + king_safety_complexity
        )
        return total_complexity

    def calculate_king_safety_evaluation(self):
        """
        Calculate the king safety evaluation based on pawn shields.

        Returns:
        - The king safety evaluation score.
        """
        pawn_shield_eval = 0
        for square, piece in self.board.piece_map().items():
            if piece.piece_type == ch.PAWN and piece.color == self.color:
                file, rank = ch.square_file(square), ch.square_rank(square)
                if rank == 1 and self.color == ch.WHITE:
                    pawn_shield_eval += 1
                elif rank == 6 and self.color == ch.BLACK:
                    pawn_shield_eval += 1
        return pawn_shield_eval

    def is_checkmate(self, move, board):
        """
        Check if a move results in a checkmate.

        Parameters:
        - move: The chess move.
        - board: The chess board.

        Returns:
        - True if the move results in a checkmate, False otherwise.
        """
        board_copy = board.copy()
        board_copy.push(move)
        return board_copy.is_checkmate

    def calculate_board_hash(self, board):
        """
        Calculate a hash value for the chess board.

        Parameters:
        - board: The chess board.

        Returns:
        - The calculated hash value.
        """
        return board.fen()

    def minimax(self, board, alpha, beta, depth):
        key = self.calculate_board_hash(board)

        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= depth:
                if entry["flag"] == "exact":
                    return entry["best_move"], entry["value"]
                elif entry["flag"] == "lowerbound":
                    alpha = max(alpha, entry["value"])
                elif entry["flag"] == "upperbound":
                    beta = min(beta, entry["value"])
                if alpha >= beta:
                    return entry["best_move"], entry["value"]

        moveList = list(board.legal_moves)
        moveList.sort(
            key=lambda move: (board.is_capture(move), board.is_check()),
            reverse=True,
        )

        if not moveList:
            return None, self.built_in_evalFunc(board)

        newCandidate = float("-inf") if depth % 2 != 0 else float("inf")
        best_move = None

        for i in moveList:
            board.push(i)
            if depth == self.maxDepth:
                value = self.built_in_evalFunc(board)
            else:
                _, value = self.minimax(board, alpha, beta, depth + 1)

            if (value > newCandidate and depth % 2 != 0) or (
                value < newCandidate and depth % 2 == 0
            ):
                best_move = i
                newCandidate = value

            if depth % 2 == 0:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)

            board.pop()

            if beta <= alpha:
                break

        flag = (
            "exact"
            if newCandidate > alpha and newCandidate < beta
            else "lowerbound" if newCandidate >= beta else "upperbound"
        )

        self.store_transposition_table_entry(
            board, self.maxDepth, newCandidate, best_move, flag
        )
        return best_move, newCandidate
