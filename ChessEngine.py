import chess as ch
import random as rd


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
        pieceValue = piece_values.get(piece_type, 0)

        if self.board.color_at(square) != self.color:
            return -pieceValue
        else:
            return pieceValue

    def engine(self):
        best_move = None
        for depth in range(1, self.maxDepth + 1):
            best_move = self.search(depth)
        return best_move

    def search(self, depth, alpha=float("-inf"), beta=float("inf")):
        best_move = None
        for move in self.board.legal_moves:
            self.board.push(move)
            value = self.minimax(depth - 1, alpha, beta, False)
            self.board.pop()

            if value > alpha:
                alpha = value
                best_move = move

            if alpha >= beta:
                break

        return best_move

    def minimax(self, depth, alpha, beta, maximizing_player):
        if depth == 0 or not any(self.board.legal_moves):
            return self.evalFunct()

        if maximizing_player:
            value = float("-inf")
            for move in self.board.legal_moves:
                self.board.push(move)
                value = max(value, self.minimax(depth - 1, alpha, beta, False))
                self.board.pop()
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:
            value = float("inf")
            for move in self.board.legal_moves:
                self.board.push(move)
                value = min(value, self.minimax(depth - 1, alpha, beta, True))
                self.board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break

        return value
