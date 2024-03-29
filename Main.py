import logging
import sys
import os
# Save the original stdout and stderr
orig_stdout = sys.stdout
orig_stderr = sys.stderr

# Redirect stdout and stderr to os.devnull
sys.stdout = open(os.devnull, "w")
sys.stderr = open(os.devnull, "w")

# Import pygame
import pygame

# Restore stdout and stderr
sys.stdout = orig_stdout
sys.stderr = orig_stderr
import chess
import MonteCarlo as MTCS
import Minimax as Minimax


class Main:
    piece_images = None

    def __init__(self, board=None):
        self.board = board if board else chess.Board()
        self.width, self.height = 500, 500
        self.square_size = self.width // 8
        self.screen = pygame.display.set_mode((self.width, self.height))
        if Main.piece_images is None:
            Main.piece_images = self.load_piece_images()
        self.selected_piece = None
        self.drag_offset = None
        self.dragging = False
        self.AI_turn = False
        self.color = "w"
        self.AI_type = "minimax"
        pygame.display.set_caption("Chess Game")

    @classmethod
    def load_piece_images(cls):
        piece_images = {}
        pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        colors = ["black", "white"]

        for piece in pieces:
            for color in colors:
                image_path = os.path.join("img", f"{piece}-{color}.png")
                piece_images[piece + ("b" if color == "black" else "w")] = (
                    pygame.image.load(image_path)
                )

        return piece_images

    def draw_board(self):
        colors = [(255, 255, 255), (0, 0, 0)]
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                pygame.draw.rect(
                    self.screen,
                    color,
                    (
                        col * self.square_size,
                        row * self.square_size,
                        self.square_size,
                        self.square_size,
                    ),
                )

    def draw_pieces(self):
        for row in range(8):
            for col in range(8):
                if self.color == "b":
                    square = chess.square(7 - col, row)  # Mirror the square for black
                else:
                    square = chess.square(col, 7 - row)
                piece = self.board.piece_at(square)
                if piece:
                    piece_name = chess.piece_name(piece.piece_type).lower()
                    piece_color = "b" if piece.color == chess.BLACK else "w"
                    piece_key = piece_name + piece_color
                    piece_image = self.piece_images[piece_key]
                    self.screen.blit(
                        piece_image, (col * self.square_size, row * self.square_size)
                    )

    def handle_mouse_click(self, event):
        logging.debug("Mouse Clicked")
        if self.dragging:
            self.dragging = False
            self.selected_piece = None
            self.drag_offset = None
            return

        x, y = event.pos
        col, row = x // self.square_size, y // self.square_size

        if 0 <= col < 8 and 0 <= row < 8:
            if self.color == "b":
                square = chess.square(7 - col, row)  # Mirror the square for black
            else:
                square = chess.square(col, 7 - row)
            piece = self.board.piece_at(square)
            if not piece:
                return
            self.selected_piece = (piece, square)
            self.drag_offset = (
                event.pos[0] - (square % 8) * self.square_size,
                event.pos[1] - (square // 8) * self.square_size,
            )
            pygame.display.flip()

    def handle_mouse_drag(self, event):
        logging.debug("Mouse Dragged")
        if not self.selected_piece:
            return

        x, y = event.pos
        new_x, new_y = x - self.drag_offset[0], y - self.drag_offset[1]
        col, row = new_x // self.square_size, new_y // self.square_size

        if 0 <= col < 8 and 0 <= row < 8:
            if self.color == "b":
                new_square_visual = chess.square(7 - col, row)
                new_x = (7 - col) * self.square_size + self.square_size / 4
                new_y = row * self.square_size + self.square_size / 4
            else:
                new_square_visual = chess.square(col, 7 - row)
                new_x = col * self.square_size + self.square_size / 4
                new_y = (7 - row) * self.square_size + self.square_size / 4

            self.selected_piece = (self.selected_piece[0], new_square_visual)
            self.drag_offset = (x - new_x, y - new_y)
            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()

    def handle_mouse_release(self, event):
        logging.debug("Mouse Released")
        if not self.dragging:
            return

        move = self.get_move_from_drag_visual(*self.selected_piece)
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            print("Invalid move!")

        self.dragging = False
        self.selected_piece = None
        self.drag_offset = None
        self.draw_board()
        self.draw_pieces()

    def get_square_at_position(self, position):
        x, y = position
        if self.color == "b":
            col = 7 - x // self.square_size
            row = y // self.square_size
        else:
            col = x // self.square_size
            row = 7 - y // self.square_size

        if 0 <= col < 8 and 0 <= row < 8:
            return chess.square(col, row)

        return None

    def get_move_from_drag(self, piece_and_square, target_square):
        piece, square = piece_and_square
        piece_type = chess.PIECE_TYPES[piece.piece_type]

        start_square = chess.square_string(square)
        target_square_str = chess.square_string(target_square)

        return chess.Move.from_uci(f"{start_square}{target_square_str}")

    def get_move_from_drag_visual(self, piece_and_square, target_square):
        piece, square = piece_and_square
        piece_type = chess.PIECE_TYPES[piece.piece_type]

        if self.color == "b":
            start_square_visual = chess.square(
                7 - square % 8, 7 - square // 8
            )  # Mirror the square for black visualization
        else:
            start_square_visual = chess.square(square % 8, square // 8)

        target_square_visual = chess.square(
            7 - target_square % 8, 7 - target_square // 8
        )  # Mirror the square for black visualization

        start_square_str_visual = chess.square_string(start_square_visual)
        target_square_str_visual = chess.square_string(target_square_visual)

        return chess.Move.from_uci(
            f"{start_square_str_visual}{target_square_str_visual}"
        )

    def play_human_move(self):
        legal_moves = [move.uci() for move in self.board.legal_moves]

        move_started = False
        start_square = None

        while not move_started:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN:
                    square = self.get_square_at_position(event.pos)
                    if square:
                        piece = self.board.piece_at(square)
                        if piece and piece.color == self.board.turn:
                            move_started = True
                            start_square = square
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        move_ended = False
        while not move_ended:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    end_square = self.get_square_at_position(event.pos)
                    if end_square:
                        move = chess.Move(start_square, end_square)
                        move_uci = move.uci()
                        if move_uci in legal_moves:
                            self.board.push_uci(move_uci)
                            move_ended = True
                        else:
                            print("Invalid move!")
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

    def play_engine_move(self, max_depth, color):
        print("engine: " + self.AI_type)
        engine = (
            Minimax.Engine(self.board, max_depth, color)
            if self.AI_type == "minimax"
            else MTCS.Engine(self.board, color)
        )

        best_move = engine.getBestMove()
        print(best_move, "best move")
        if best_move in self.board.legal_moves:
            self.board.push(best_move)
        else:
            print("Engine made an illegal move!")

    def start_game(self):
        logging.basicConfig(level=logging.DEBUG)

        self.select_side_screen()

        ai_color = "w" if self.color == "b" else "b"
        self.AI_turn = False if self.color == "w" else True
        max_depth = 7  # Set the initial max depth for the engine

        clock = pygame.time.Clock()

        while not self.board.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_click(event)
                elif event.type == pygame.MOUSEMOTION and self.selected_piece:
                    self.handle_mouse_drag(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_release(event)

            self.draw_board()

            self.draw_pieces()
            pygame.display.flip()

            ## TEST AI MTCS VS AI Minimax ##
            print("The engine is thinking...")

            self.play_engine_move(max_depth, ai_color)

            self.AI_type = "monte_carlo" if self.AI_type == "minimax" else "minimax"

            ai_color = "w" if ai_color == "b" else "b"

            ## HUMAN VS AI ##
            # if self.AI_turn:
            #     print("The engine is thinking...")
            #     self.play_engine_move(max_depth, ai_color, self.AI_type)
            #     self.AI_turn = False
            # else:
            #     self.play_human_move()
            #     self.AI_turn = True

        # Game over, show end game screen
        winner = "White" if self.board.turn == chess.BLACK else "Black"
        print("Looser: " + self.AI_type)
        print("Winner: " + "monte_carlo" if self.AI_type == "minimax" else "minimax")
        if self.end_game_screen(winner):
            return True  # Start a new game

        else:
            pygame.quit()

        clock.tick(60)  # Limit frames per second

        pygame.quit()

    def end_game_screen(self, winner):
        font = pygame.font.Font(None, 36)
        text = font.render(f"Winner: {winner}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))

        restart_button = pygame.Rect(self.width // 2 - 75, self.height // 2, 150, 50)

        pygame.draw.rect(self.screen, (255, 255, 255), restart_button)

        restart_text = font.render("Restart", True, (0, 0, 0))
        restart_text_rect = restart_text.get_rect(center=restart_button.center)

        while True:
            self.screen.fill((0, 0, 0))
            self.screen.blit(text, text_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), restart_button)
            self.screen.blit(restart_text, restart_text_rect)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if restart_button.collidepoint(event.pos):
                        return True  # Restart the game
            pygame.time.Clock().tick(60)

    def select_side_screen(self):
        font = pygame.font.Font(None, 36)
        text = font.render("Select Your Side and AI Type:", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))

        white_button = pygame.Rect(self.width // 4 - 75, self.height // 2, 150, 50)
        black_button = pygame.Rect(3 * self.width // 4 - 75, self.height // 2, 150, 50)
        minimax_button = pygame.Rect(
            self.width // 4 - 75, self.height // 2 + 75, 150, 50
        )
        monte_carlo_button = pygame.Rect(
            self.width // 2.5 + 75, self.height // 2 + 75, 200, 50
        )

        while True:
            self.screen.fill((0, 0, 0))
            self.screen.blit(text, text_rect)

            # Change the color of the buttons based on the selected AI type
            minimax_color = (
                (0, 255, 0) if self.AI_type == "minimax" else (255, 255, 255)
            )
            monte_carlo_color = (
                (0, 255, 0) if self.AI_type == "monte_carlo" else (10, 10, 10)
            )
            minimax_button = (
                pygame.Rect(self.width // 4 - 75, self.height // 2 + 75, 170, 60)
                if self.AI_type == "minimax"
                else pygame.Rect(self.width // 4 - 75, self.height // 2 + 75, 150, 50)
            )
            monte_carlo_button = (
                pygame.Rect(self.width // 2.5 + 75, self.height // 2 + 75, 220, 60)
                if self.AI_type == "monte_carlo"
                else pygame.Rect(self.width // 2.5 + 75, self.height // 2 + 75, 200, 50)
            )

            pygame.draw.rect(self.screen, (255, 255, 255), white_button)
            pygame.draw.rect(self.screen, (10, 10, 10), black_button)
            pygame.draw.rect(self.screen, minimax_color, minimax_button)
            pygame.draw.rect(self.screen, monte_carlo_color, monte_carlo_button)

            white_text = font.render("White", True, (0, 0, 0))
            white_text_rect = white_text.get_rect(center=white_button.center)

            black_text = font.render("Black", True, (255, 255, 255))
            black_text_rect = black_text.get_rect(center=black_button.center)

            minimax_text = font.render("Minimax AI", True, (0, 0, 0))
            minimax_text_rect = minimax_text.get_rect(center=minimax_button.center)

            monte_carlo_text = font.render("Monte Carlo AI", True, (255, 255, 255))
            monte_carlo_text_rect = monte_carlo_text.get_rect(
                center=monte_carlo_button.center
            )

            self.screen.blit(white_text, white_text_rect)
            self.screen.blit(black_text, black_text_rect)
            self.screen.blit(minimax_text, minimax_text_rect)
            self.screen.blit(monte_carlo_text, monte_carlo_text_rect)

            pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if white_button.collidepoint(event.pos):
                        self.color = "w"
                        return
                    elif black_button.collidepoint(event.pos):
                        self.color = "b"
                        return
                    elif minimax_button.collidepoint(event.pos):
                        self.AI_type = "minimax"
                    elif monte_carlo_button.collidepoint(event.pos):
                        self.AI_type = "monte_carlo"


# Create an instance and start a game
if __name__ == "__main__":
    pygame.init()
    game = Main()
    start_game = True
    while start_game:
        start_game = game.start_game()
