import chess as ch
import random as rd
import os
import concurrent.futures
import copy
from collections import namedtuple
import json
import threading


class ChessEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ch.Move):
            return {"san": obj.uci()}
        elif isinstance(obj, MoveEvaluation):
            return {"value": obj.value, "move": obj.move.uci() if obj.move else None}
        return super().default(obj)


# Define a named tuple to represent a move and its evaluation
MoveEvaluation = namedtuple("MoveEvaluation", ["value", "move"])

# Piece values for evaluation
piece_values = {
    ch.PAWN: 1,
    ch.ROOK: 5.1,
    ch.BISHOP: 3.33,
    ch.KNIGHT: 3.2,
    ch.QUEEN: 8.8,
    ch.KING: 999,
}


class Engine:
    def __init__(
        self, board, maxDepth, color, cache_file="./cache/transposition_cache.json"
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
        self.transposition_table = {}
        self.transposition_table_lock = threading.Lock()
        self.cache_file = cache_file
        self.load_cache()

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
        except (FileNotFoundError, json.JSONDecodeError):
            with open(self.cache_file, "w") as file:
                file.write(
                    "{}"
                )  # Write an empty JSON object if the file is newly created or contains invalid data

    def update_cache(self, board, depth, value, best_move, flag):
        """
        Update the transposition cache with the latest information.

        Parameters:
        - board: The chess board.
        - depth: The depth of the search.
        - value: The evaluation value.
        - best_move: The best move found.
        - flag: The flag indicating the type of result (exact, lowerbound, upperbound).
        """
        key = self.calculate_board_hash(board)

        # Make a copy of the transposition table before modifying it
        with self.transposition_table_lock:
            transposition_copy = self.transposition_table.copy()

        if key not in transposition_copy or depth >= transposition_copy[key]["depth"]:
            serialized_best_move = best_move.uci() if best_move is not None else None

            # Update the copy of the transposition table
            transposition_copy[key] = {
                "depth": depth,
                "value": value,
                "best_move": serialized_best_move,
                "flag": flag,
            }

            # Save the updated transposition table to the cache file
            with open(self.cache_file, "w") as file:
                json.dump(transposition_copy, file, cls=ChessEncoder)

    def getBestMove(self):
        """
        Get the best move using the engine.

        Returns:
        - The best move found by the engine.
        """
        return self.engine()

    def evalFunct(self):
        """
        Evaluate the current position based on material and positional factors.

        Returns:
        - The evaluation score for the current position.
        """
        compt = 0
        # Sums up the material values
        for square in self.board.piece_map():
            compt += self.squareResPoints(square)
        compt += self.mateOpportunity() + self.openning() + 0.01 * rd.random()
        return compt

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
                return -999
            else:
                return 999
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

    def calculate_material_evaluation(self):
        """
        Calculate the material evaluation based on the piece values.

        Returns:
        - The material evaluation score.
        """
        material_eval = 0
        for square, piece in self.board.piece_map().items():
            value = piece_values.get(piece.piece_type, 0)
            if piece.color == self.color:
                material_eval += value
            else:
                material_eval -= value
        return material_eval

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

    def squareResPoints(self, square):
        """
        Calculate the evaluation points for a given square.

        Parameters:
        - square: The chess square.

        Returns:
        - The evaluation points for the specified square.
        """
        piece_type = self.board.piece_type_at(square)
        piece_value = piece_values.get(piece_type, 0)

        if self.board.color_at(square) != self.color:
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

        return self.search_parallel(self.maxDepth)

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

    def calculate_time_limit(self, complexity):
        """
        Calculate the time limit for the engine search.

        Parameters:
        - complexity: The complexity of the position.

        Returns:
        - The calculated time limit for the search.
        """
        base_time_limit = 15
        complexity_threshold = 10  # Adjust this threshold based on testing
        time_limit = base_time_limit + max(0, (complexity - complexity_threshold) / 10)
        return time_limit

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

    def quiescenceSearch(self, board_copy, alpha, beta, depth=0):
        """
        Perform quiescence search to handle tactical positions.

        Parameters:
        - board_copy: Copy of the current chess board.
        - alpha: Alpha value for alpha-beta pruning.
        - beta: Beta value for alpha-beta pruning.
        - depth: Current depth in the search.

        Returns:
        - The evaluation score and the best move found.
        """
        if depth >= 2:
            return self.evalFunct(), None

        stand_pat = self.evalFunct()
        if stand_pat >= beta:
            return beta, None
        if alpha < stand_pat:
            alpha = stand_pat

        best_move = None
        for move in board_copy.legal_moves:
            if board_copy.is_capture(move) or self.is_checkmate(move, board_copy):
                board_copy.push(move)
                score = -self.quiescenceSearch(board_copy, -beta, -alpha, depth + 1)[0]

                board_copy.pop()

                if score >= beta:
                    return beta, None
                if score > alpha:
                    alpha = score
                    best_move = move

        return alpha, best_move

    def calculate_board_hash(self, board):
        """
        Calculate a hash value for the chess board.

        Parameters:
        - board: The chess board.

        Returns:
        - The calculated hash value.
        """
        return board.fen()

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

    def minimax(self, board, depth, alpha, beta, maximizing_player, color):
        """
        Perform minimax search with alpha-beta pruning.

        Parameters:
        - board: The chess board.
        - depth: The current depth in the search.
        - alpha: Alpha value for alpha-beta pruning.
        - beta: Beta value for alpha-beta pruning.
        - maximizing_player: True if maximizing player's turn, False otherwise.
        - color: The color of the engine.

        Returns:
        - The evaluation score and the best move found.
        """
        stack = [(board, depth, alpha, beta, maximizing_player, color)]
        best_move = None

        while stack:
            board, depth, alpha, beta, maximizing_player, color = stack.pop()
            key = self.calculate_board_hash(board)
            if key in self.transposition_table:
                entry = self.transposition_table[key]
                if entry["depth"] >= depth:
                    if entry["flag"] == "exact":
                        return entry["value"], entry["best_move"]
                    elif entry["flag"] == "lowerbound":
                        alpha = max(alpha, entry["value"])
                    elif entry["flag"] == "upperbound":
                        beta = min(beta, entry["value"])
                    if alpha >= beta:
                        return entry["value"], entry["best_move"]

            if depth == 0 or board.is_game_over():
                return self.quiescenceSearch(board, alpha, beta)

            if maximizing_player:
                max_eval = -99999
                best_move = None
                for move in board.legal_moves:
                    board.push(move)
                    evaluation = self.minimax(
                        board, depth - 1, alpha, beta, False, color
                    )

                    if type(evaluation) == float:
                        evaluation = (evaluation, None)

                    eval = evaluation[0]
                    board.pop()
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break

                self.store_transposition_table_entry(
                    board, depth, max_eval, best_move, "exact"
                )

                self.update_cache(board, depth, max_eval, best_move, "exact")

                return max_eval, best_move

            else:
                min_eval = 99999
                best_move = None
                for move in board.legal_moves:
                    board.push(move)
                    evaluation = self.minimax(
                        board, depth - 1, alpha, beta, True, color
                    )
                    if type(evaluation) == float:
                        evaluation = (evaluation, None)
                    eval = evaluation[0]
                    board.pop()
                    if eval < min_eval:
                        min_eval = eval
                        best_move = move
                    beta = min(beta, eval)
                    if beta <= alpha:
                        break

                self.store_transposition_table_entry(
                    board, depth, min_eval, best_move, "exact"
                )
                self.update_cache(board, depth, min_eval, best_move, "exact")
                return min_eval, best_move

    def search(
        self, depth, alpha=-99999, beta=99999
    ):  # currently unused (using the search with parallelization)
        """
        Perform a simple search without parallelization.

        Parameters:
        - depth: The depth of the search.
        - alpha: Alpha value for alpha-beta pruning.
        - beta: Beta value for alpha-beta pruning.

        Returns:
        - The best move found by the search.
        """
        best_move = None
        moves = list(self.board.legal_moves)

        moves.sort(
            key=lambda move: self.board.is_capture(move) or self.board.is_check()
        )

        for move in moves:
            self.board.push(move)
            move_value = self.minimax(depth - 1, alpha, beta, False, self.color)
            self.board.pop()
            if move_value > alpha:
                alpha = move_value
                best_move = move
            if alpha >= beta:
                break
        return best_move

    def search_parallel(self, depth, alpha=-99999, beta=99999):
        """
        Perform parallelized search using parallelization for better performance.

        Parameters:
        - depth: The depth of the search.
        - alpha: Alpha value for alpha-beta pruning.
        - beta: Beta value for alpha-beta pruning.

        Returns:
        - The best move found by the parallelized search.
        """
        best_move = None
        moves = list(self.board.legal_moves)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.evaluate_move,
                    copy.deepcopy(self.board),
                    move,
                    depth,
                    alpha,
                    beta,
                ): move
                for move in moves
            }
            for future in concurrent.futures.as_completed(futures):
                (move_value, move) = future.result()
                if move_value > alpha:
                    alpha = move_value
                    best_move = move
                if alpha >= beta:
                    break
        return best_move

    def evaluate_move(self, board_copy, move, depth, alpha, beta):
        """
        Evaluate a move using minimax within a parallelized context.

        Parameters:
        - board_copy: Copy of the current chess board.
        - move: The chess move to be evaluated.
        - depth: The current depth in the search.
        - alpha: Alpha value for alpha-beta pruning.
        - beta: Beta value for alpha-beta pruning.

        Returns:
        - A named tuple with the evaluation score and the move.
        """
        board_copy.push(move)
        move_value, _ = self.minimax(
            board_copy, depth - 1, -beta, -alpha, False, self.color
        )

        if move_value == -99999:
            return MoveEvaluation(value=-99999, move=move)
        if type(move_value) == float:
            move_value = (move_value, None)
        return MoveEvaluation(value=move_value[0], move=move)
