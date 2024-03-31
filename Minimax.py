import random
from time import sleep
import chess as ch
import os
from collections import OrderedDict, namedtuple
import json
import logging
import chess.engine
import stable


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
        stockfish,
        cache_file="./cache/transposition_cache_Minimax.json",
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
        self.engine_eval = stockfish
        self.openings_folder = "./oppenings"
        self.openings = {}
        self.load_openings()

    def store_transposition_table_entry(self, board, depth, value, best_move, flag):
        """
        Store an entry in the transposition table.

        Parameters:
        - board: The chess board.
        - depth: The depth of the search.
        - value: The evaluation value.
        - best_move: The best move found.
        - flag: the flag indicating if a move was evaluated by stockfish

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
        - flag: the flag indicating if a move was evaluated by stockfish
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

        if len(self.board.move_stack) < 20:
            return self.play_opening_move(self.board)

        if self.engine_eval is not None:
            result = self.engine_eval.play(self.board, chess.engine.Limit(time=2))
            return result.move
        return self.engine()

    def evalFunct(self, board):
        """
        Evaluate the current position based on material and positional factors.

        Returns:
        - The evaluation score for the current position.
        """
        if self.engine_eval is None:
            return self.built_in_evalFunc(board)
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
            return self.evalFunct(board)

    def calculate_repetition_penalty(self, move, board):
        """
        Calculate a penalty for move repetition.

        Parameters:
        - move: The chess move.

        Returns:
        - A penalty if the move has been repeated, 0 otherwise.
        """
        move_history = [board_move.uci() for board_move in board.move_stack]
        move_count = move_history.count(move.uci())

        # Apply a penalty if the move has been repeated more than once
        if move_count > 1:
            return -100 * move_count
        else:
            return 0

    def load_openings(self):
        """
        Load openings from PGN files in the specified folder and its subfolders.
        """
        self.openings = {}
        for root, dirs, files in os.walk(self.openings_folder):
            for filename in files:
                if filename.endswith(".pgn"):
                    opening_name = os.path.splitext(filename)[0]
                    full_path = os.path.join(root, filename)
                    self.openings[opening_name] = chess.pgn.read_game(open(full_path))
        # Shuffle the keys of the openings dictionary
        keys = list(self.openings.keys())
        random.shuffle(keys)
        self.openings = {key: self.openings[key] for key in keys}

    def play_opening_move(self, board):
        """
        Play an opening move from the loaded openings.

        Parameters:
        - board: The current chess board.

        Returns:
        - The updated board after playing the move.
        - The move played.
        """
        for opening_name, opening in self.openings.items():
            newBoard = chess.Board()
            for move in opening.mainline_moves():
                if newBoard == board:
                    sleep(1)
                    print(move)
                    return move
                newBoard.push(move)

        return self.engine()

    def getPieceValue(self, piece, x, y):

        if piece == None:
            return 0

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

        absolute_value = getAbsoluteValue(piece, piece.color == "w", x, y)
        return absolute_value if piece.color == chess.WHITE else -absolute_value

    def built_in_evalFunc(self, board):
        a = self.mateOpportunity(board)
        if a is not None:
            return a

        compt = 0
        # Sums up the material values
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                compt += self.getPieceValue(board.piece_at(square), i, j)

        compt *= 2

        def has_castled(board, color):
            for move in board.move_stack:
                if (
                    board.is_castling(move)
                    and board.color_at(move.from_square) == color
                ):
                    return True
            return False

        # Then in your evaluation function:
        if has_castled(board, self.color):
            compt += 100  # Give 100 points if the AI has castled

        last_move = board.peek()
        compt += self.openning(board) + self.calculate_repetition_penalty(
            last_move, board
        )

        if board.is_check():
            if board.turn == self.color:
                compt -= 50
            else:
                compt += 50

        compt += random.randint(-1, 1)
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

            return -999999

        if board.is_game_over():
            return 0
        return None

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
            return 0
            opening_evaluation = self.calculate_opening_evaluation(board)
            return opening_evaluation

    def calculate_opening_evaluation(self, board):
        """
        Calculate the overall opening evaluation based on material, pawn structure, and mobility.

        Returns:
        - The opening evaluation score.
        """
        return (
            0.2 * self.calculate_material_evaluation(board)
            + 0.3 * self.calculate_pawn_structure_evaluation(board)
            + 0.2 * self.calculate_mobility_evaluation(board)
            + 0.3 * self.calculate_king_safety_evaluation(board)
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

    def deepening_search_depth(self):
        """
        Increase the search depth as the game progresses.

        This function adjusts the maximum depth for the search algorithm based on the number of pieces on the board.
        """
        # Calculate the number of pieces on the board
        num_pieces = len(self.board.piece_map())

        # Adjust the maximum depth based on the number of pieces
        if num_pieces < 5:
            self.maxDepth += 6
        elif num_pieces < 8:
            self.maxDepth += 5
        elif num_pieces < 10:
            self.maxDepth += 4
        elif num_pieces < 13:
            self.maxDepth += 3
        elif num_pieces < 15:
            self.maxDepth += 2
        elif num_pieces < 20:
            self.maxDepth += 1
        print(self.maxDepth)

    def engine(self):
        """
        Perform the main engine search.

        Returns:
        - The best move found by the engine.
        """
        key = self.calculate_board_hash(self.board)

        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= self.maxDepth:
                if (entry["flag"] == True and self.engine_eval is not None) or (
                    entry["flag"] == False and self.engine_eval is None
                ):
                    return chess.Move.from_uci(entry["best_move"]["san"])

        if self.engine_eval is None:
            self.deepening_search_depth()

        move, val = self.minimax(self.board.copy(), float("-inf"), float("inf"), 1)

        self.store_transposition_table_entry(
            self.board,
            self.maxDepth,
            val,
            move,
            True if self.engine_eval is not None else False,
        )

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

    def calculate_king_safety_evaluation(self, board):
        """
        Calculate the king safety evaluation based on pawn shields.

        Returns:
        - The king safety evaluation score.
        """
        pawn_shield_eval = 0
        for square, piece in board.piece_map().items():
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

        moveList = list(board.legal_moves)
        moveList.sort(
            key=lambda move: (board.is_capture(move), board.is_check()),
            reverse=True,
        )

        if not moveList:
            return None, self.evalFunct(board)

        newCandidate = float("-inf") if depth % 2 != 0 else float("inf")
        best_move = None

        for i in moveList:
            board.push(i)
            if depth == self.maxDepth:
                value = self.evalFunct(board)
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

        return best_move, newCandidate
