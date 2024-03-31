import math
import chess
import random

import chess.engine


class MonteCarloTreeSearchAI:
    def __init__(self, stockfish):
        self.engine = stockfish

    def get_best_move(self, board):
        root_node = Node(board)
        for _ in range(1000):  # Number of simulations
            node = root_node
            temp_board = board.copy()
            while not node.is_terminal():
                if not node.is_fully_expanded():
                    return self.expand(node).move
                node = self.select(node)
                temp_board.push(node.move)
            result = self.simulate(temp_board)
            self.backpropagate(node, result)
        return root_node.get_best_move().move



    def select(self, node):
        ucb_scores = [
            child.wins / child.visits
            + 1.4 * math.sqrt(math.log(node.visits) / child.visits)
            for child in node.children
        ]
        selected_child = node.children[ucb_scores.index(max(ucb_scores))]
        return selected_child

    def evaluate(self, board):
        # Evaluate the board position and return a score
        # You can use heuristics such as piece values, piece square tables, mobility, etc.
        # Here's an example that considers material points, king safety, pawn structure, and control of the center:
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
        }

        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                if piece.color == chess.WHITE:
                    score += piece_values[piece.piece_type]
                    if square in center_squares:
                        score += 0.5  # Bonus for controlling the center
                    if piece.piece_type == chess.KING:
                        # Evaluate king safety
                        king_safety_score = self.evaluate_king_safety(board, square)
                        score += king_safety_score
                else:
                    score -= piece_values[piece.piece_type]
                    if square in center_squares:
                        score -= 0.5  # Penalty for opponent controlling the center
                    if piece.piece_type == chess.KING:
                        # Evaluate opponent king safety
                        king_safety_score = self.evaluate_king_safety(board, square)
                        score -= king_safety_score
        return score

    def evaluate_king_safety(self, board, king_square):
        # Evaluate king safety based on factors such as pawn shield, pawn attacks, and open files
        pawn_shield_squares = [
            chess.A2,
            chess.B2,
            chess.C2,
            chess.D2,
            chess.E2,
            chess.F2,
            chess.G2,
            chess.H2,
        ]
        pawn_shield_score = 0

        for square in pawn_shield_squares:
            if board.piece_at(square) is not None:
                pawn_shield_score += 0.1  # Bonus for having pawns in the shield squares
        pawn_attack_squares = [
            chess.A3,
            chess.B3,
            chess.C3,
            chess.D3,
            chess.E3,
            chess.F3,
            chess.G3,
            chess.H3,
        ]
        pawn_attack_score = 0
        for square in pawn_attack_squares:
            if (
                board.piece_at(square) is not None
                and board.piece_at(square).color != board.turn
            ):
                pawn_attack_score += (
                    0.1  # Penalty for opponent pawns attacking the king
                )
        open_files = 0
        for file in chess.FILES:
            if board.is_file_open(file):
                open_files += 1  # Bonus for open files near the king
        king_safety_score = pawn_shield_score + pawn_attack_score + open_files
        return king_safety_score

    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            if result == node.board.turn:
                node.wins += 1
            else:
                node.wins -= 1
            node = node.parent
        pass


class Node:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def is_terminal(self):
        # Check if the node represents a terminal state (e.g., checkmate, stalemate)
        return self.board.is_checkmate() or self.board.is_stalemate()

    def is_fully_expanded(self):
        # Check if all possible moves from the node have been expanded
        return len(self.children) == len(list(self.board.legal_moves))

    def get_best_move(self):
        # Return the best move based on the node's statistics
        best_child = max(self.children, key=lambda child: child.visits)

        return best_child.move
