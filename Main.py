import logging
import sys
import pygame
import chess
import ChessEngine as ce
import os

pygame.init()


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

        square = self.get_square_at_position(event.pos)
        if not square:
            return
        piece = self.board.piece_at(square)
        if not piece:
            return
        self.selected_piece = (piece, square)
        self.drag_offset = (
            event.pos[0] - (square % 8) * self.square_size,
            event.pos[1] - (square // 8) * self.square_size,
        )
        pygame.display.flip()

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
        engine = ce.Engine(self.board, max_depth, color)
        best_move = engine.getBestMove()
        if best_move in self.board.legal_moves:
            self.board.push(best_move)
        else:
            print("Engine made an illegal move!")

    def start_game(self):
        logging.basicConfig(level=logging.DEBUG)

        self.select_side_screen()

        ai_color = "w" if self.color == "b" else "b"
        self.AI_turn = False if self.color == "w" else True
        max_depth = 4  # Set the initial max depth for the engine

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

            # ## TEST AI VS AI ##
            # print("The engine is thinking...")
            # self.play_engine_move(max_depth, ai_color)  ##
            # ai_color = "w" if ai_color == "b" else "b"  ##

            if self.AI_turn:
                print("The engine is thinking...")
                self.play_engine_move(max_depth, ai_color)
                self.AI_turn = False
            else:
                self.play_human_move()
                self.AI_turn = True

        clock.tick(60)  # Limit frames per second

        pygame.quit()

    def select_side_screen(self):
        font = pygame.font.Font(None, 36)
        text = font.render("Select Your Side:", True, (255, 255, 255))
        text_rect = text.get_rect(center=(self.width // 2, self.height // 2 - 50))

        white_button = pygame.Rect(self.width // 4 - 75, self.height // 2, 150, 50)
        black_button = pygame.Rect(3 * self.width // 4 - 75, self.height // 2, 150, 50)

        pygame.draw.rect(self.screen, (255, 255, 255), white_button)
        pygame.draw.rect(self.screen, (0, 0, 0), black_button)

        white_text = font.render("White", True, (0, 0, 0))
        white_text_rect = white_text.get_rect(center=white_button.center)

        black_text = font.render("Black", True, (255, 255, 255))
        black_text_rect = black_text.get_rect(center=black_button.center)

        while True:
            self.screen.fill((0, 0, 0))
            self.screen.blit(text, text_rect)
            pygame.draw.rect(self.screen, (255, 255, 255), white_button)
            pygame.draw.rect(self.screen, (10, 10, 10), black_button)
            self.screen.blit(white_text, white_text_rect)
            self.screen.blit(black_text, black_text_rect)
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


# Create an instance and start a game
game = Main()
game.start_game()
