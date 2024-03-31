import math
import os
from collections import OrderedDict, namedtuple
import json
import logging
import random as rd
import chess
import chess.engine


class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.score = 0.0


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

# Piece values for evaluation
piece_values = {
    chess.PAWN: 1,
    chess.ROOK: 5,
    chess.BISHOP: 3,
    chess.KNIGHT: 3,
    chess.QUEEN: 8,
    chess.KING: 999,
}


class Engine:
    def __init__(
        self,
        board,
        color,
        # cache_file="E:/chess_cache/transposition_cache.json",
        stockfish,
        iterations=16,
        cache_file="./cache/transposition_cache_MTCS.json",
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
        self.transposition_table = OrderedDict()
        self.cache_file = cache_file
        self.iterations = iterations
        self.load_cache()
        self.engine_eval = stockfish

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
        if self.engine_eval is None:
            return self.built_in_evalFunct(board)

        try:
            result = self.engine_eval.analyse(board, chess.engine.Limit(time=0.5))
            score = result["score"].relative.score()
            if not isinstance(score, (int, float, complex)):
                print("error getting stockfish score")
                score = 0
            return score
        except (ValueError, TimeoutError) as e:
            logging.error("Error parsing Stockfish output.")
            logging.error("Retying to get the value for the board.")
            return self.stockfish_evalFunct(board)

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
                return 9999999
            else:
                return -9999999
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
            0.4 * self.calculate_material_evaluation(board)
            + 0.1 * self.calculate_pawn_structure_evaluation(board)
            + 0.1 * self.calculate_mobility_evaluation(board)
            + 0.4 * self.calculate_king_safety_evaluation(board)
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
            if piece.color == board.turn:
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
            if piece.piece_type == chess.PAWN:
                file, rank = chess.square_file(square), chess.square_rank(square)
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
        piece_value = piece_values[piece_type]

        if board.color_at(square) != self.color:
            return -piece_value
        else:
            if piece_type == chess.PAWN:
                file, rank = chess.square_file(square), chess.square_rank(square)
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
        # move = self.iterative_deepening_search()
        root_node = Node(self.board)
        move = self.mcts_search(root_node, self.iterations)

        return move

    def calculate_king_safety_evaluation(self, board):
        """
        Calculate the king safety evaluation based on pawn shields.

        Returns:
        - The king safety evaluation score.
        """
        pawn_shield_eval = 0
        for square, piece in board.piece_map().items():
            if piece.piece_type == chess.PAWN and piece.color == self.color:
                file, rank = chess.square_file(square), chess.square_rank(square)
                if rank == 1 and self.color == chess.WHITE:
                    pawn_shield_eval += 1
                elif rank == 6 and self.color == chess.BLACK:
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

    def mcts_search(self, root, iterations):

        key = self.calculate_board_hash(self.board)

        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= self.iterations:
                if (entry["flag"] == True and self.engine_eval is not None) or (
                    entry["flag"] == False and self.engine_eval == None
                ):
                    return chess.Move.from_uci(entry["best_move"]["san"])

        for i in range(1, iterations + 1):
            selected_node = self.selection(root)
            expanded_node = self.expansion(selected_node)
            simulation_result = self.simulation(expanded_node)
            self.backpropagation(expanded_node, simulation_result, i)

        # After all iterations, select the best move based on statistics
        best_child = max(root.children, key=lambda child: child.visits)
        self.update_transposition_table(self.board, best_child, self.iterations)
        self.update_cache()
        return best_child.state.peek()  # Return the move from the best child

    def selection(self, node):
        while node.children:
            node = max(node.children, key=lambda child: ucb1(child))
        return node

    def expansion(self, node):
        legal_moves = list(node.state.legal_moves)

        if legal_moves:
            # Evaluate each legal move
            move_evaluations = [
                (move, self.eval_move(node.state, move)) for move in legal_moves
            ]

            # Sort moves based on evaluation score (higher score is better)
            sorted_moves = sorted(move_evaluations, key=lambda x: x[1], reverse=True)

            # Choose the best move for expansion
            best_move, _ = sorted_moves[0]

            new_state = node.state.copy()
            new_state.push(best_move)

            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
            return new_node
        else:
            return node  # No legal moves, return the same node

    def backpropagation(self, node, result, iteration):
        while node:
            node.visits += 1
            node.score += result
            node = node.parent

    def check_transposition_table(self, state):
        key = self.calculate_board_hash(state)
        return key in self.transposition_table

    def update_transposition_table(self, state, result, iteration):
        key = self.calculate_board_hash(state)

        if (
            key not in self.transposition_table
            or result > self.transposition_table[key]["value"]
        ):
            self.transposition_table[key] = {
                "iterations": iteration,
                "best_move": result,
                "flag": True if self.engine_eval is not None else False,
            }

    def simulation(self, node):
        # Perform a simple random playout
        sim_board = node.state.copy()
        while not sim_board.is_game_over():
            sim_moves = list(sim_board.legal_moves)
            sim_move = rd.choice(sim_moves)
            sim_board.push(sim_move)
        return self.stockfish_evalFunct(sim_board)

    def eval_move(self, board, move):
        """
        Evaluate a move based on your custom criteria.
        Adjust this function according to your evaluation strategy.
        """
        board.push(move)

        value = self.stockfish_evalFunct(board)

        board.pop()
        return value


def ucb1(node, exploration_weight=1.0):
    if node.visits == 0:
        return math.inf  # Ensure exploration for unvisited nodes
    exploitation = node.score / node.visits
    exploration = exploration_weight * math.sqrt(
        math.log(node.parent.visits) / node.visits
    )
    return exploitation + exploration
