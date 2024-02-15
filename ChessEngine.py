import time
import chess as ch
import random as rd
import os
from collections import OrderedDict, namedtuple
import json
import logging
import stable as s


stable = s.stable()


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
        cache_file="./cache/transposition_cache.json",
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

    def evalFunct(self, board):

        if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
        if board.is_stalemate():
            return 0
        if board.is_insufficient_material():
            return 0

        wp = len(board.pieces(ch.PAWN, ch.WHITE))
        bp = len(board.pieces(ch.PAWN, ch.BLACK))
        wn = len(board.pieces(ch.KNIGHT, ch.WHITE))
        bn = len(board.pieces(ch.KNIGHT, ch.BLACK))
        wb = len(board.pieces(ch.BISHOP, ch.WHITE))
        bb = len(board.pieces(ch.BISHOP, ch.BLACK))
        wr = len(board.pieces(ch.ROOK, ch.WHITE))
        br = len(board.pieces(ch.ROOK, ch.BLACK))
        wq = len(board.pieces(ch.QUEEN, ch.WHITE))
        bq = len(board.pieces(ch.QUEEN, ch.BLACK))

        material = (
            piece_values[ch.PAWN] * (wp - bp)
            + piece_values[ch.KNIGHT] * (wn - bn)
            + piece_values[ch.BISHOP] * (wb - bb)
            + piece_values[ch.ROOK] * (wr - br)
            + piece_values[ch.QUEEN] * (wq - bq)
        )

        pawnsq = sum([stable.pawnstable[i] for i in board.pieces(ch.PAWN, ch.WHITE)])
        pawnsq = pawnsq + sum(
            [
                -stable.pawnstable[ch.square_mirror(i)]
                for i in board.pieces(ch.PAWN, ch.BLACK)
            ]
        )
        knightsq = sum(
            [stable.knightstable[i] for i in board.pieces(ch.KNIGHT, ch.WHITE)]
        )
        knightsq = knightsq + sum(
            [
                -stable.knightstable[ch.square_mirror(i)]
                for i in board.pieces(ch.KNIGHT, ch.BLACK)
            ]
        )
        bishopsq = sum(
            [stable.bishopstable[i] for i in board.pieces(ch.BISHOP, ch.WHITE)]
        )
        bishopsq = bishopsq + sum(
            [
                -stable.bishopstable[ch.square_mirror(i)]
                for i in board.pieces(ch.BISHOP, ch.BLACK)
            ]
        )
        rooksq = sum([stable.rookstable[i] for i in board.pieces(ch.ROOK, ch.WHITE)])
        rooksq = rooksq + sum(
            [
                -stable.rookstable[ch.square_mirror(i)]
                for i in board.pieces(ch.ROOK, ch.BLACK)
            ]
        )
        queensq = sum([stable.queenstable[i] for i in board.pieces(ch.QUEEN, ch.WHITE)])
        queensq = queensq + sum(
            [
                -stable.queenstable[ch.square_mirror(i)]
                for i in board.pieces(ch.QUEEN, ch.BLACK)
            ]
        )
        kingsq = sum([stable.kingstable[i] for i in board.pieces(ch.KING, ch.WHITE)])
        kingsq = kingsq + sum(
            [
                -stable.kingstable[ch.square_mirror(i)]
                for i in board.pieces(ch.KING, ch.BLACK)
            ]
        )

        eval = (
            material
            + pawnsq
            + knightsq
            + bishopsq
            + rooksq
            + queensq
            + kingsq
            + self.mateOpportunity()
            + self.openning()
        )

        print(board)
        if board.turn:
            print(eval)
            return eval
        else:
            print(-eval)
            return -eval

    # def evalFunct(self):
    #     """
    #     Evaluate the current position based on material and positional factors.

    #     Returns:
    #     - The evaluation score for the current position.
    #     """
    #     compt = 0
    #     # Sums up the material values
    #     for square in self.board.piece_map():
    #         compt += self.squareResPoints(square)
    #     compt += self.mateOpportunity() + self.openning() + 0.01 * rd.random()
    #     return compt

    def mateOpportunity(self):
        """
        Check if the current position presents a checkmate opportunity.

        Returns:
        - A large positive score if the current player has a checkmate opportunity.
        - A large negative score if the opponent has a checkmate opportunity.
        - Otherwise, returns 0.
        """
        if self.board.is_checkmate():
            if self.board.turn == self.color:
                return -99999
            else:
                return 999999
        else:
            return 0

    def openning(self):
        """
        Evaluate the opening phase of the game.

        Returns:
        - A score based on the number of legal moves, adjusted for the opening phase.
        """
        legal_moves = list(self.board.legal_moves)
        if self.board.fullmove_number < 10:
            if self.board.turn == self.color:
                return 1 / 30 * len(legal_moves)
            else:
                return -1 / 30 * len(legal_moves)
        else:
            opening_evaluation = self.calculate_opening_evaluation()
            return opening_evaluation

    def calculate_opening_evaluation(self):
        """
        Calculate the overall opening evaluation based on material, pawn structure, and mobility.

        Returns:
        - The opening evaluation score.
        """
        return (
            0.5 * self.calculate_material_evaluation()
            + 0.3 * self.calculate_pawn_structure_evaluation()
            + 0.2 * self.calculate_mobility_evaluation()
        )

    # def calculate_material_evaluation(self):
    #     """
    #     Calculate the material evaluation based on the piece values.

    #     Returns:
    #     - The material evaluation score.
    #     """
    #     material_eval = 0
    #     for square, piece in self.board.piece_map().items():
    #         value = piece_values.get(piece.piece_type, 0)
    #         if piece.color == self.color:
    #             material_eval += value
    #         else:
    #             material_eval -= value
    #     return material_eval

    def calculate_pawn_structure_evaluation(self):
        """
        Calculate the pawn structure evaluation.

        Returns:
        - The pawn structure evaluation score.
        """
        pawn_structure_eval = 0
        for square, piece in self.board.piece_map().items():
            if piece.piece_type == ch.PAWN and piece.color == self.color:
                file, rank = ch.square_file(square), ch.square_rank(square)
                pawn_structure_eval += 0.1 * (4 - abs(3 - file))
        return pawn_structure_eval

    def calculate_mobility_evaluation(self):
        """
        Calculate the mobility evaluation based on legal moves.

        Returns:
        - The mobility evaluation score.
        """
        mobility_eval = 0
        for move in self.board.legal_moves:
            if self.board.turn == self.color:
                mobility_eval += 1  # Increase for own legal moves
            else:
                mobility_eval -= 1  # Decrease for opponent's legal moves
        return mobility_eval

    # def squareResPoints(self, square):
    #     """
    #     Calculate the evaluation points for a given square.

    #     Parameters:
    #     - square: The chess square.

    #     Returns:
    #     - The evaluation points for the specified square.
    #     """
    #     piece_type = self.board.piece_type_at(square)
    #     piece_value = piece_values.get(piece_type, 0)

    #     if self.board.color_at(square):
    #         return -piece_value
    #     else:
    #         if piece_type == ch.PAWN:
    #             file, rank = ch.square_file(square), ch.square_rank(square)
    #             pawn_structure_value = 0.1 * (4 - abs(3 - file))
    #             return piece_value + pawn_structure_value
    #         else:
    #             return piece_value

    def engine(self):
        """
        Perform the main engine search.

        Returns:
        - The best move found by the engine.
        """
        move = self.iterative_deepening_search()
        self.update_cache()
        return move

    def iterative_deepening_search(self):
        move, _ = self.minimax(self.board.copy(), float("-inf"), float("inf"), 1)

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
                    return i, entry["value"]
                elif entry["flag"] == "lowerbound":
                    alpha = max(alpha, entry["value"])
                elif entry["flag"] == "upperbound":
                    beta = min(beta, entry["value"])
                if alpha >= beta:
                    return i, entry["value"]

        # get list of legal moves of the current position
        moveList = list(board.legal_moves)
        moveList.sort(
            key=lambda move: (self.board.is_capture(move), self.board.is_check()),
            reverse=True,
        )

        # If there are no legal moves left, return None and the evaluation function
        if not moveList:
            return None, self.evalFunct(board)

        # initialise newCandidate and best_move
        newCandidate = float("-inf") if depth % 2 != 0 else float("inf")
        best_move = None

        # analyse board after deeper moves
        for i in moveList:

            board.push(i)

            # Get value of move i (by exploring the repercussions)
            if depth == self.maxDepth:
                value = self.evalFunct(board)
            else:
                _, value = self.minimax(board, alpha, beta, depth + 1)

            # Basic minmax algorithm:
            # if maximizing
            if value > newCandidate and depth % 2 != 0:
                best_move = i
                newCandidate = value
            # if minimizing
            elif value < newCandidate and depth % 2 == 0:
                best_move = i
                newCandidate = value

            # Alpha-beta pruning cuts:
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

        # Update transposition table
        self.store_transposition_table_entry(
            self.board, self.maxDepth, newCandidate, best_move, flag
        )

        # Return result
        return best_move, newCandidate
