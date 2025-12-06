import logging
import random
import sys
import os
from time import sleep

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
from Minimax_improved import MinimaxAI  # Use improved version

# Try to import the new unified model loader, fallback to legacy
try:
    from model_loader import load_chess_model, list_available_models, ChessModelWrapper
    NEW_MODEL_LOADER = True
except ImportError:
    from pytorch_model import PytorchModel
    from old_model import OldPytorchModel
    NEW_MODEL_LOADER = False
    print("Warning: Using legacy model loader")


class Main:
    piece_images = None

    def __init__(
        self,
        board=None,
        stockfish_path="./stockfish/stockfish-windows-x86-64-avx2.exe",
    ):
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
        self.AI_type = "minimax"  # Default AI type
        self.neural_model_type = "limited"  # Default neural model
        pygame.display.set_caption("Chess Game")
        
        # Fix typo: oppenings -> openings
        self.openings_folder = "./openings"
        if not os.path.exists(self.openings_folder):
            self.openings_folder = "./oppenings"  # Fallback to old name
        
        self.openings = {}
        self.load_openings()
        
        # Lazy load models - only load what's needed
        self._neural_engine = None
        self._minimax_engine = None
        
        # Show selection screen
        self.select_side_screen()

        # New flags for human input
        self.human_moved = False
        self.start_click_square = None
        self.start_click_pos = None
        self.drag_started = False
        # New flags for two-click support
        self.waiting_for_second_click = False
        self.ignore_release = False
        
        # Parallel search settings
        self.use_parallel_minimax = True  # Enable Lazy SMP by default
        self.minimax_depth = 6
    
    @property
    def neural_engine(self):
        """Lazy load neural engine only when needed."""
        if self._neural_engine is None:
            print(f"Loading {self.neural_model_type} neural network...")
            if NEW_MODEL_LOADER:
                self._neural_engine = load_chess_model(self.neural_model_type)
            else:
                self._neural_engine = PytorchModel()
        return self._neural_engine
    
    @property
    def minimax_engine(self):
        """Lazy load minimax engine only when needed."""
        if self._minimax_engine is None:
            ai_color = "w" if self.color == "b" else "b"
            self._minimax_engine = MinimaxAI(
                self.openings, 
                ai_color, 
                depth=self.minimax_depth,
                use_parallel=self.use_parallel_minimax
            )
        return self._minimax_engine


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
        x, y = event.pos
        col, row = x // self.square_size, y // self.square_size

        if not (0 <= col < 8 and 0 <= row < 8):
            self.selected_piece = None
            self.dragging = False
            self.start_click_square = None
            self.waiting_for_second_click = False
            return

        if self.color == "b":
            square = chess.square(7 - col, row)  
        else:
            square = chess.square(col, 7 - row)


        # If no piece is selected yet, try to select an allied piece.
        if not self.selected_piece:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                print(f"Selected piece at {chess.square_name(square)}")
                self.selected_piece = (piece, square)
                self.waiting_for_second_click = True
            # Else: do nothing (click on empty or enemy square does nothing)
        else:
            # There is an active selection (waiting for second click)
            # If the clicked square is different than the originally selected square,
            # treat it as the move destination (for two-click mode).
            if self.waiting_for_second_click and square != self.selected_piece[1]:
                move = chess.Move(self.selected_piece[1], square)
                if move in self.board.legal_moves:
                    print(f"Valid move (2-click): {move} ({chess.square_name(self.selected_piece[1])} -> {chess.square_name(square)})")
                    self.board.push(move)
                    self.human_moved = True
                else:
                    print(f"Invalid move attempted: {move}")
                    print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
                # Clear selection and disable processing on release.
                self.selected_piece = None
                self.waiting_for_second_click = False
                self.ignore_release = True
            else:
                # If the same square is clicked, update selection.
                piece = self.board.piece_at(square)
                if piece and piece.color == self.board.turn:
                    print(f"Switching selection to {chess.square_name(square)}")
                    self.selected_piece = (piece, square)
        pygame.display.flip()

    def handle_mouse_motion(self, event):
        if self.selected_piece and self.start_click_pos:
            dx = event.pos[0] - self.start_click_pos[0]
            dy = event.pos[1] - self.start_click_pos[1]
            dist = (dx**2 + dy**2)**0.5
            # If motion is significant, consider it a drag.
            if dist > 5:
                self.drag_started = True
                self.dragging = True
                # Cancel two-click mode if dragging
                self.waiting_for_second_click = False
            # (You might add visual feedback for dragging here.)
    
    def handle_mouse_release(self, event):
        if self.ignore_release:
            self.ignore_release = False
            return

        if not self.selected_piece:
            return

        x, y = event.pos
        col, row = x // self.square_size, y // self.square_size
        if not (0 <= col < 8 and 0 <= row < 8):
            self.selected_piece = None
            self.dragging = False
            self.start_click_square = None
            return

        if self.color == "b":
            release_square = chess.square(7 - col, row)  # Only mirror the column for black
        else:
            release_square = chess.square(col, 7 - row)


        # If a drag was detected, use the release square as the destination.
        # Otherwise, in two-click mode, only act if the release square
        # is different from the originally clicked square.
        if self.drag_started:
            dest_square = release_square
        else:
            # Two-click mode: if the release square is different from the selected square,
            # treat it as the destination.
            if release_square != self.selected_piece[1]:
                dest_square = release_square
            else:
                # Clicking the same square does nothing.
                self.selected_piece = None
                self.start_click_square = None
                return

        move = chess.Move(self.selected_piece[1], dest_square)
        if move in self.board.legal_moves:
            print(f"Valid move: {move} ({chess.square_name(self.selected_piece[1])} -> {chess.square_name(dest_square)})")
            self.board.push(move)
            self.human_moved = True
        else:
            print(f"Invalid move attempted: {move}")
            print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")

        # Clear selection and drag info.
        self.selected_piece = None
        self.dragging = False
        self.start_click_square = None
        self.drag_started = False
        pygame.display.flip()

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

    def get_move_from_drag_visual(self, piece, target_square):
        piece, from_square = piece
        
        # Both squares are already in internal board representation
        # No need for complex transformations
        move = chess.Move(from_square, target_square)
        
        # Debug info
        print(f"Attempting move: {move} ({chess.square_name(from_square)} to {chess.square_name(target_square)})")
        print(f"Legal moves: {[m.uci() for m in self.board.legal_moves]}")
        
        return move



    def push_move(self, best_move):
        if best_move in self.board.legal_moves:
            self.board.push(best_move)
        else:
            print("Engine made an illegal move!")

    def play_engine_move(self, max_depth, color):
        """Play a move using the selected AI engine."""
        print(f"Engine: {self.AI_type}")
        
        if self.AI_type == "minimax":
            print("Minimax AI thinking...")
            best_move = self.minimax_engine.get_best_move(self.board)
            self.push_move(best_move)
            print(f"MINIMAX MOVE: {best_move}")
        
        elif self.AI_type == "neural_mcts":
            print(f"Neural MCTS ({self.neural_model_type}) thinking...")
            # Use MCTS for stronger play
            if NEW_MODEL_LOADER:
                best_move = self.neural_engine.get_best_move(
                    self.board, 
                    method="mcts",
                    num_simulations=200,  # Adjust based on hardware
                    temperature=0.1
                )
            else:
                best_move = self.neural_engine.get_best_move_mcts(self.board)
            self.push_move(best_move)
            print(f"NEURAL MCTS MOVE: {best_move}")
        
        elif self.AI_type == "neural_direct":
            print(f"Neural Direct ({self.neural_model_type}) thinking...")
            # Fast direct policy (weaker but instant)
            if NEW_MODEL_LOADER:
                best_move = self.neural_engine.get_best_move(
                    self.board,
                    method="direct",
                    temperature=0.5
                )
            else:
                best_move = self.neural_engine.best_move_direct(self.board)
            self.push_move(best_move)
            print(f"NEURAL DIRECT MOVE: {best_move}")
        
        else:
            # Default to minimax
            print("Unknown AI type, using minimax...")
            best_move = self.minimax_engine.get_best_move(self.board)
            self.push_move(best_move)
            print(f"MOVE: {best_move}")

    def start_game(self):
        logging.basicConfig(level=logging.DEBUG)

        ai_color = "w" if self.color == "b" else "b"

        # self.AI_turn = False  
        max_depth = 3  # Set the initial max depth for the engine

        clock = pygame.time.Clock()

        while not self.board.is_game_over():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                # Process mouse events only if it is human's turn.
                if not self.AI_turn:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.handle_mouse_click(event)
                    elif event.type == pygame.MOUSEMOTION:
                        self.handle_mouse_motion(event)
                    elif event.type == pygame.MOUSEBUTTONUP:
                        self.handle_mouse_release(event)

            self.draw_board()
            self.draw_pieces()
            pygame.display.flip()

            ## AI VS AI ##
            # self.human_moved = True
            
            # print("The engine is thinking...")
            # self.play_engine_move(max_depth, ai_color)
            # self.AI_turn = not self.AI_turn


            ## AI VS HUMAN ##
            # If it's human turn and a move has been made, then switch to AI turn.
            if not self.AI_turn and self.human_moved:
                self.AI_turn = True
                self.human_moved = False

            # If it's AI's turn, execute the engine move and then switch turn.
            if self.AI_turn:
                print("The engine is thinking...")
                sleep(1)
                self.play_engine_move(max_depth, ai_color)
                self.AI_turn = False

            clock.tick(60)  # Limit frames per second

        # Game over, show end game screen
        print(self.board.outcome)
        winner = "White" if self.board.turn == chess.BLACK else "Black"
        print("Looser: " + self.AI_type)
        print("Winner: " + ("monte_carlo" if self.AI_type == "minimax" else "minimax"))
        print(str(chess.pgn.Game.from_board(self.board)))
        if self.end_game_screen(winner):
            return True  # Start a new game
        else:
            pygame.quit()
            return False

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
        """Selection screen for side, AI type, and model."""
        font = pygame.font.Font(None, 32)
        small_font = pygame.font.Font(None, 24)
        
        # Available AI types
        ai_types = ["minimax", "neural_mcts", "neural_direct"]
        ai_labels = ["Minimax", "Neural MCTS", "Neural Direct"]
        
        # Available neural model types
        model_types = ["limited", "small", "medium", "big"]
        model_labels = ["Limited (2GB)", "Small", "Medium", "Big"]
        
        selected_ai_idx = 0  # Default to minimax
        selected_model_idx = 0  # Default to limited
        
        while True:
            self.screen.fill((30, 30, 40))
            
            # Title
            title = font.render("Chess AI Setup", True, (255, 255, 255))
            self.screen.blit(title, (self.width // 2 - title.get_width() // 2, 20))
            
            # Section 1: Side selection
            side_text = small_font.render("Choose Your Side:", True, (200, 200, 200))
            self.screen.blit(side_text, (20, 60))
            
            white_button = pygame.Rect(20, 90, 100, 40)
            black_button = pygame.Rect(130, 90, 100, 40)
            
            white_color = (100, 200, 100) if self.color == "w" else (80, 80, 80)
            black_color = (100, 200, 100) if self.color == "b" else (40, 40, 40)
            
            pygame.draw.rect(self.screen, white_color, white_button, border_radius=5)
            pygame.draw.rect(self.screen, black_color, black_button, border_radius=5)
            pygame.draw.rect(self.screen, (255, 255, 255), white_button, 2, border_radius=5)
            pygame.draw.rect(self.screen, (255, 255, 255), black_button, 2, border_radius=5)
            
            white_txt = small_font.render("White", True, (0, 0, 0) if self.color == "w" else (200, 200, 200))
            black_txt = small_font.render("Black", True, (255, 255, 255))
            self.screen.blit(white_txt, (white_button.centerx - white_txt.get_width()//2, white_button.centery - 8))
            self.screen.blit(black_txt, (black_button.centerx - black_txt.get_width()//2, black_button.centery - 8))
            
            # Section 2: AI Type selection
            ai_text = small_font.render("Choose AI Type:", True, (200, 200, 200))
            self.screen.blit(ai_text, (20, 150))
            
            ai_buttons = []
            for i, label in enumerate(ai_labels):
                btn = pygame.Rect(20 + i * 155, 180, 145, 40)
                ai_buttons.append(btn)
                
                color = (100, 200, 100) if i == selected_ai_idx else (60, 60, 70)
                pygame.draw.rect(self.screen, color, btn, border_radius=5)
                pygame.draw.rect(self.screen, (150, 150, 150), btn, 1, border_radius=5)
                
                txt = small_font.render(label, True, (0, 0, 0) if i == selected_ai_idx else (200, 200, 200))
                self.screen.blit(txt, (btn.centerx - txt.get_width()//2, btn.centery - 8))
            
            # Section 3: Neural Model selection (only show if neural AI selected)
            if selected_ai_idx > 0:  # Neural AI selected
                model_text = small_font.render("Choose Neural Model:", True, (200, 200, 200))
                self.screen.blit(model_text, (20, 240))
                
                model_buttons = []
                for i, label in enumerate(model_labels):
                    btn = pygame.Rect(20 + i * 118, 270, 110, 35)
                    model_buttons.append(btn)
                    
                    color = (100, 150, 200) if i == selected_model_idx else (60, 60, 70)
                    pygame.draw.rect(self.screen, color, btn, border_radius=5)
                    pygame.draw.rect(self.screen, (150, 150, 150), btn, 1, border_radius=5)
                    
                    txt = small_font.render(label, True, (0, 0, 0) if i == selected_model_idx else (180, 180, 180))
                    self.screen.blit(txt, (btn.centerx - txt.get_width()//2, btn.centery - 6))
            else:
                model_buttons = []
            
            # AI description
            desc_y = 330
            descriptions = {
                "minimax": "Traditional search with alpha-beta pruning. Depth 6. Strong tactical play.",
                "neural_mcts": "Neural network + MCTS search. 200 simulations. Best quality.",
                "neural_direct": "Direct neural policy output. Fast but less accurate.",
            }
            desc = descriptions.get(ai_types[selected_ai_idx], "")
            desc_lines = [desc[i:i+55] for i in range(0, len(desc), 55)]
            for line in desc_lines:
                desc_txt = small_font.render(line, True, (150, 150, 150))
                self.screen.blit(desc_txt, (20, desc_y))
                desc_y += 20
            
            # Start button
            start_button = pygame.Rect(self.width // 2 - 75, self.height - 80, 150, 50)
            pygame.draw.rect(self.screen, (50, 150, 50), start_button, border_radius=8)
            pygame.draw.rect(self.screen, (100, 255, 100), start_button, 2, border_radius=8)
            
            start_txt = font.render("START", True, (255, 255, 255))
            self.screen.blit(start_txt, (start_button.centerx - start_txt.get_width()//2, start_button.centery - 10))
            
            pygame.display.flip()
            
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    
                    # Side buttons
                    if white_button.collidepoint(pos):
                        self.color = "w"
                        self.AI_turn = False
                    elif black_button.collidepoint(pos):
                        self.color = "b"
                        self.AI_turn = True
                    
                    # AI type buttons
                    for i, btn in enumerate(ai_buttons):
                        if btn.collidepoint(pos):
                            selected_ai_idx = i
                    
                    # Model buttons
                    for i, btn in enumerate(model_buttons):
                        if btn.collidepoint(pos):
                            selected_model_idx = i
                    
                    # Start button
                    if start_button.collidepoint(pos):
                        self.AI_type = ai_types[selected_ai_idx]
                        self.neural_model_type = model_types[selected_model_idx]
                        print(f"Selected: Side={self.color}, AI={self.AI_type}, Model={self.neural_model_type}")
                        return
            
            pygame.time.Clock().tick(60)


# Create an instance and start a game
if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    pygame.init()
    start_game = True
    while start_game:
        game = Main()
        start_game = game.start_game()