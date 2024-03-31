from collections import OrderedDict, namedtuple
import copy
import json
import logging
import os
import random
import sys
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
    def __init__(self, board, depth, color):
        self.board = board
        self.max_depth = depth
        self.transposition_table = OrderedDict()
        self.color = color
        self.is_maximizing_player = True if self.color == "black" else False
        self.openings_folder = "./oppenings"
        cache_file = "./cache/transposition_cache_Minimaxv2.json"
        self.cache_file = cache_file
        self.load_cache()
        self.openings = {}
        self.load_openings()

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
        # Shuffle the keys of the dictionary
        opening_names = list(self.openings.keys())
        random.shuffle(opening_names)
        self.openings = {name: self.openings[name] for name in opening_names}

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

        key = self.calculate_board_hash(self.board)

        if key in self.transposition_table:
            entry = self.transposition_table[key]
            if entry["depth"] >= self.max_depth:
                return chess.Move.from_uci(entry["best_move"]["san"])

        self.deepening_search_depth()

        move = self.minimax_root()
        self.store_transposition_table_entry(
            self.board,
            self.max_depth,
            move,
        )

        self.update_cache()
        return move

    def minimax_root(self):
        newGameMoves = [
            (move, self.score_move(move, self.board)) for move in self.board.legal_moves
        ]

        newGameMoves.sort(key=lambda x: x[1], reverse=True)


        # Create a multiprocessing pool
        pool = Pool(processes=cpu_count())

        # Create a list to hold the results
        results = []

        for move, _ in newGameMoves:
            # Create a new copy of the board for each move
            board = copy.deepcopy(self.board)
            board.push(move)
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
        self, board, alpha, beta, is_maximizing_player, depth=0, max_depth=7
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
        if depth <= 0:
            return -self.quiescence(board, beta, alpha, is_maximizing_player)

        # Null move pruning
        if depth >= 7 and not board.is_check():
            board.push(chess.Move.null())
            score = -self.minimax(
                depth - 3, board, -beta, -beta + 1, not is_maximizing_player
            )
            board.pop()
            if score >= beta:
                return beta

        # Get all legal moves
        legal_moves = list(board.legal_moves)

        # Score each move using a heuristic
        newGameMoves = [(move, self.score_move(move, board)) for move in legal_moves]

        # Sort the moves in descending order of their scores
        newGameMoves.sort(key=lambda x: x[1], reverse=True)

        if is_maximizing_player:
            bestMove = -9999
            for move, _ in newGameMoves:
                board.push(move)
                bestMove = max(
                    bestMove,
                    self.minimax(
                        depth - 1, board, alpha, beta, not is_maximizing_player
                    ),
                )
                board.pop()
                alpha = max(alpha, bestMove)
                if beta <= alpha:
                    break
        else:
            bestMove = 9999
            for move, _ in newGameMoves:
                board.push(move)
                bestMove = min(
                    bestMove,
                    self.minimax(
                        depth - 1, board, alpha, beta, not is_maximizing_player
                    ),
                )
                board.pop()
                beta = min(beta, bestMove)
                if beta <= alpha:
                    break
        return bestMove

    def score_move(self, move, board):
        if move is not None:
            board.push(move)
        eval = 0
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    eval += self.getPieceValue(piece, i, j)
        if move is not None:
            board.pop()
        return eval

    def evaluateBoard(self, board):
        eval = self.score_move(None, board)
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7 - i)
                piece = board.piece_at(square)
                if piece is not None:
                    # Get the basic value of the piece
                    piece_value = self.getPieceValue(piece, i, j)
                    eval += piece_value

                    # Add a bonus for piece mobility
                    legal_moves = len(list(board.legal_moves))
                    eval += 0.1 * piece_value * legal_moves

                    # Add a bonus for control of the center in early moves
                    if (
                        i >= 3
                        and i <= 4
                        and j >= 3
                        and j <= 4
                        and len(board.move_stack) < 10
                    ):
                        eval += 0.2 * piece_value

                    # Add a penalty for king exposure
                    if piece.piece_type == chess.KING:
                        king_exposure = len(board.attackers(not piece.color, square))
                        eval -= 0.5 * piece_value * king_exposure

                    # Add a bonus for pawn structure
                    if piece.piece_type == chess.PAWN:
                        # Bonus for passed pawns
                        if all(piece.piece_type != chess.PAWN for k in range(i + 1, 8)):
                            eval += 0.5 * piece_value
                        # Penalty for isolated pawns
                        if (j == 0 or piece.piece_type != chess.PAWN) and (
                            j == 7 or piece.piece_type != chess.PAWN
                        ):
                            eval -= 0.5 * piece_value

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
