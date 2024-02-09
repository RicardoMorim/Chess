import chess as ch
import random as rd
import time
import concurrent.futures
import copy
from collections import namedtuple

MoveEvaluation = namedtuple("MoveEvaluation", ["value", "move"])

piece_values = {
    ch.PAWN: 1,
    ch.ROOK: 5.1,
    ch.BISHOP: 3.33,
    ch.KNIGHT: 3.2,
    ch.QUEEN: 8.8,
    ch.KING: 999,
}


class Engine:
    def __init__(self, board, maxDepth, color):
        self.board = board
        self.color = color
        self.maxDepth = maxDepth
        self.transposition_table = {}

    def getBestMove(self):

        return self.engine()

    def evalFunct(self):
        compt = 0
        # Sums up the material values
        for square in self.board.piece_map():
            compt += self.squareResPoints(square)
        compt += self.mateOpportunity() + self.openning() + 0.01 * rd.random()
        return compt

    def mateOpportunity(self):
        if self.board.turn == self.color:
            return -999
        else:
            return 999

    def openning(self):
        legal_moves = list(self.board.legal_moves)
        if self.board.fullmove_number < 10:
            if self.board.turn == self.color:
                return 1 / 30 * len(legal_moves)
            else:
                return -1 / 30 * len(legal_moves)
        else:
            return 0

    def squareResPoints(self, square):
        piece_type = self.board.piece_type_at(square)
        piece_value = piece_values.get(piece_type, 0)

        if self.board.color_at(square) != self.color:
            return -piece_value
        else:
            # Adjust pawn structure evaluation
            if piece_type == ch.PAWN:
                file, rank = ch.square_file(square), ch.square_rank(square)
                pawn_structure_value = 0.1 * (
                    4 - abs(3 - file)
                )  # Example adjustment, adjust as needed
                return piece_value + pawn_structure_value
            else:
                return piece_value

    def engine(self, time_limit=15):
        best_move = None

        start_time = time.time()
        for depth in range(1, self.maxDepth + 1):
            if time.time() - start_time > time_limit:
                break  # Exit the loop if time limit is reached
            best_move = self.search_parallel(depth)

        return best_move

    def quiescenceSearch(self, board_copy, alpha, beta, depth=0):
        if depth >= 2:  # Set a limit to the quiescence search depth
            return self.evalFunct(), None

        stand_pat = self.evalFunct()
        if stand_pat >= beta:
            return beta, None
        if alpha < stand_pat:
            alpha = stand_pat

        best_move = None
        for move in board_copy.legal_moves:
            if board_copy.is_capture(move) or board_copy.is_check():
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
        return board.fen()

    def store_transposition_table_entry(self, board, depth, value, best_move, flag):
        key = self.calculate_board_hash(board)
        self.transposition_table[key] = {
            "depth": depth,
            "value": value,
            "best_move": best_move,
            "flag": flag,
        }

    def minimax(self, board, depth, alpha, beta, maximizing_player, color):

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
                evaluation = self.minimax(board, depth - 1, alpha, beta, False, color)

                if type(evaluation) == float:
                    evaluation = (evaluation, None)

                eval = evaluation[0]
                board.pop()  # Undo move
                if eval > max_eval:  # Compare only the score
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

            self.store_transposition_table_entry(
                board, depth, max_eval, best_move, "exact"
            )

            return max_eval, best_move

        else:
            min_eval = 99999
            best_move = None
            for move in board.legal_moves:
                board.push(move)
                evaluation = self.minimax(board, depth - 1, alpha, beta, True, color)
                if type(evaluation) == float:
                    evaluation = (evaluation, None)
                eval = evaluation[0]
                board.pop()  # Undo move
                if eval < min_eval:  # Compare only the score
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break
                
            self.store_transposition_table_entry(
                board, depth, min_eval, best_move, "exact"
            )

            return min_eval, best_move

    def search(self, depth, alpha=-99999, beta=99999):
        best_move = None
        moves = list(self.board.legal_moves)

        # Sort the moves based on capturing moves and checks
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
                move_value, move = future.result()
                if move_value > alpha:
                    alpha = move_value
                    best_move = move
                if alpha >= beta:
                    break

        return best_move

    def evaluate_move(self, board_copy, move, depth, alpha, beta):
        board_copy.push(move)
        move_value, _ = self.minimax(
            board_copy, depth - 1, -beta, -alpha, False, self.color
        )

        if move_value == -99999:
            return MoveEvaluation(value=-99999, move=move)
        if type(move_value) == float:
            move_value = (move_value, None)
        return MoveEvaluation(value=move_value[0], move=move)
