import logging
import random
import sys
import os
from time import sleep
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import chess
from Minimax_improved import MinimaxAI  

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
        self.width, self.height = 512, 512
        self.square_size = self.width // 8
        
        # Create Tkinter window
        self.root = tk.Tk()
        self.root.title("Chess Game")
        self.root.resizable(False, False)
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height)
        self.canvas.pack()
        
        if Main.piece_images is None:
            Main.piece_images = self.load_piece_images()
        self.selected_piece = None
        self.drag_offset = None
        self.dragging = False
        self.AI_turn = False
        self.color = "w"
        self.AI_type = "minimax"  # Default AI type
        self.neural_model_type = "limited"  # Default neural model
        
        # Fix typo: oppenings -> openings
        self.openings_folder = "./openings"
        if not os.path.exists(self.openings_folder):
            self.openings_folder = "./oppenings"  # Fallback to old name
        
        self.openings = {}
        self.load_openings()
        
        # Lazy load models - only load what's needed
        self._neural_engine = None
        self._minimax_engine = None

        self.running = True
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
        self.white_minimax_depth = 6
        self.black_minimax_depth = 6
        
        # AI vs AI mode settings (initialized before selection screen)
        self.ai_vs_ai_mode = False
        self.white_ai_type = "minimax"
        self.white_model_type = "limited"
        self.black_ai_type = "neural_mcts"
        self.black_model_type = "limited"
        self._white_neural_engine = None
        self._black_neural_engine = None
        self._white_minimax_engine = None
        self._black_minimax_engine = None
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.handle_mouse_click)
        self.canvas.bind("<B1-Motion>", self.handle_mouse_motion)
        self.canvas.bind("<ButtonRelease-1>", self.handle_mouse_release)
        
        # Show selection screen (this will set ai_vs_ai_mode if selected)
        self.select_side_screen()
    
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
    
    def get_ai_engine(self, ai_type, model_type, color, depth=None):
        """Get or create an AI engine for the specified type and color."""
        if ai_type == "minimax":
            # Create minimax engine for specified color
            cache_key = f"_minimax_{color}"
            if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
                # Use provided depth or fetch from color-specific depth
                use_depth = depth if depth is not None else self.minimax_depth
                engine = MinimaxAI(
                    self.openings, 
                    color, 
                    depth=use_depth,
                    use_parallel=self.use_parallel_minimax
                )
                setattr(self, cache_key, engine)
            return getattr(self, cache_key)
        else:
            # Neural engine (same model can be used for both colors)
            cache_key = f"_neural_{model_type}"
            if not hasattr(self, cache_key) or getattr(self, cache_key) is None:
                print(f"Loading {model_type} neural network (variant: {model_type})...")
                if NEW_MODEL_LOADER:
                    engine = load_chess_model(model_type)
                else:
                    engine = PytorchModel()
                setattr(self, cache_key, engine)
            return getattr(self, cache_key)

    @classmethod
    def load_piece_images(cls):
        piece_images = {}
        pieces = ["pawn", "knight", "bishop", "rook", "queen", "king"]
        colors = ["black", "white"]

        for piece in pieces:
            for color in colors:
                image_path = os.path.join("img", f"{piece}-{color}.png")
                img = Image.open(image_path)
                img = img.resize((64, 64), Image.Resampling.LANCZOS)
                piece_images[piece + ("b" if color == "black" else "w")] = ImageTk.PhotoImage(img)

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
        colors = ["#F0D9B5", "#B58863"]  # Light and dark squares
        for row in range(8):
            for col in range(8):
                color = colors[(row + col) % 2]
                x1 = col * self.square_size
                y1 = row * self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

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
                    x = col * self.square_size
                    y = row * self.square_size
                    self.canvas.create_image(x + self.square_size // 2, y + self.square_size // 2, 
                                            image=piece_image)

    def handle_mouse_click(self, event):
        x, y = event.x, event.y
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
        self.redraw()

    def handle_mouse_motion(self, event):
        if self.selected_piece and self.start_click_pos:
            dx = event.x - self.start_click_pos[0]
            dy = event.y - self.start_click_pos[1]
            dist = (dx**2 + dy**2)**0.5
            # If motion is significant, consider it a drag.
            if dist > 5:
                self.drag_started = True
                self.dragging = True
                # Cancel two-click mode if dragging
                self.waiting_for_second_click = False
    
    def handle_mouse_release(self, event):
        if self.ignore_release:
            self.ignore_release = False
            return

        if not self.selected_piece:
            return

        x, y = event.x, event.y
        col, row = x // self.square_size, y // self.square_size
        if not (0 <= col < 8 and 0 <= row < 8):
            self.selected_piece = None
            self.dragging = False
            self.start_click_square = None
            return

        if self.color == "b":
            release_square = chess.square(7 - col, row)
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
        self.redraw()

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
                    num_simulations=1000,  # Adjust based on hardware
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
                    temperature=0.1
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

    def play_ai_vs_ai_move(self):
        """Play a move in AI vs AI mode."""
        # Determine which AI is playing
        if self.board.turn == chess.WHITE:
            ai_type = self.white_ai_type
            model_type = self.white_model_type
            depth = self.white_minimax_depth
            color = "w"
            label = "WHITE"
        else:
            ai_type = self.black_ai_type
            model_type = self.black_model_type
            depth = self.black_minimax_depth
            color = "b"
            label = "BLACK"
        
        print(f"{label} ({ai_type}) thinking...")
        
        engine = self.get_ai_engine(ai_type, model_type, color, depth=depth)
        
        if ai_type == "minimax":
            best_move = engine.get_best_move(self.board)
        elif ai_type == "neural_mcts":
            if NEW_MODEL_LOADER:
                best_move = engine.get_best_move(
                    self.board, 
                    method="mcts",
                    num_simulations=1000,
                    temperature=0.1
                )
            else:
                best_move = engine.get_best_move_mcts(self.board)
        elif ai_type == "neural_direct":
            if NEW_MODEL_LOADER:
                best_move = engine.get_best_move(
                    self.board,
                    method="direct",
                    temperature=0.1
                )
            else:
                best_move = engine.best_move_direct(self.board)
        else:
            best_move = engine.get_best_move(self.board)
        
        print(f"{label} plays: {best_move}")
        self.push_move(best_move)
    
    def redraw(self):
        """Redraw the entire board and pieces."""
        self.canvas.delete("all")
        self.draw_board()
        self.draw_pieces()
    
    def game_loop(self):
        """Main game loop using Tkinter's after method."""
        if not self.board.is_game_over() and self.running:
            self.redraw()
            
            ## AI VS AI MODE ##
            if self.ai_vs_ai_mode:
                self.play_ai_vs_ai_move()
                self.root.after(500, self.game_loop)  # Delay to see moves
            
            ## AI VS HUMAN MODE ##
            else:
                # If it's human turn and a move has been made, then switch to AI turn.
                if not self.AI_turn and self.human_moved:
                    self.AI_turn = True
                    self.human_moved = False

                # If it's AI's turn, execute the engine move and then switch turn.
                if self.AI_turn:
                    print("The engine is thinking...")
                    self.root.after(100, self._ai_move_async)
                else:
                    self.root.after(100, self.game_loop)
        elif self.board.is_game_over():
            self.game_over()
    
    def _ai_move_async(self):
        """Execute AI move asynchronously."""
        ai_color = "w" if self.color == "b" else "b"
        max_depth = 6
        self.play_engine_move(max_depth, ai_color)
        self.AI_turn = False
        self.root.after(100, self.game_loop)
    
    def game_over(self):
        """Handle game over."""
        print(self.board.outcome)
        winner = "White" if self.board.turn == chess.BLACK else "Black"
        print(str(chess.pgn.Game.from_board(self.board)))
        self.end_game_screen(winner)

    def end_game_screen(self, winner):
        result = messagebox.askyesno("Game Over", f"Winner: {winner}\n\nPlay again?")
        if result:
            self.root.destroy()
            # Restart will be handled by main
        else:
            self.root.destroy()
            sys.exit()

    def start_game(self):
        logging.basicConfig(level=logging.DEBUG)
        self.running = True
        self.game_loop()
        self.root.mainloop()
        return False  # No restart for now

    def select_side_screen(self):
        """Selection screen for side, AI type, and model."""
        # Create a new toplevel window for selection
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Chess AI Setup")
        selection_window.geometry("600x700")
        selection_window.resizable(False, False)
        selection_window.grab_set()  # Make it modal
        
        # Set color scheme
        BG_COLOR = "#1e1e2e"
        FG_COLOR = "#ffffff"
        ACCENT_COLOR = "#4CAF50"
        SECONDARY_COLOR = "#FF9800"
        LABEL_COLOR = "#b0b0b0"
        
        selection_window.configure(bg=BG_COLOR)
        
        # State variables
        game_mode = tk.StringVar(value="vs_human")
        selected_ai = tk.StringVar(value="minimax")
        selected_model = tk.StringVar(value="baseline")
        selected_depth = tk.IntVar(value=6)
        white_ai = tk.StringVar(value="neural_mcts")
        white_model = tk.StringVar(value="baseline")
        white_depth = tk.IntVar(value=6)
        black_ai = tk.StringVar(value="minimax")
        black_model = tk.StringVar(value="baseline")
        black_depth = tk.IntVar(value=6)
        
        def update_human_ui(*args):
            """Update UI visibility based on selected AI type."""
            for widget in model_frame.winfo_children():
                widget.destroy()
            for widget in depth_frame.winfo_children():
                widget.destroy()
            
            if selected_ai.get() == "minimax":
                tk.Label(depth_frame, text="Depth:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10)).pack(side="left", padx=5)
                tk.Scale(depth_frame, from_=1, to=8, orient="horizontal", variable=selected_depth,
                        bg=SECONDARY_COLOR, fg=FG_COLOR, length=200).pack(side="left", padx=5)
            else:
                tk.Label(model_frame, text="Model Variant:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10)).pack(anchor="w", padx=5)
                for variant in ["baseline", "attack", "est"]:
                    tk.Radiobutton(model_frame, text=variant.capitalize(), variable=selected_model, value=variant,
                                 bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR, 
                                 activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                                 font=("Arial", 9)).pack(anchor="w", padx=20, pady=2)
        
        def update_white_ui(*args):
            """Update White AI UI."""
            for widget in white_model_frame.winfo_children():
                widget.destroy()
            for widget in white_depth_frame.winfo_children():
                widget.destroy()
            
            if white_ai.get() == "minimax":
                tk.Label(white_depth_frame, text="Depth:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 9)).pack(side="left", padx=5)
                tk.Scale(white_depth_frame, from_=1, to=8, orient="horizontal", variable=white_depth,
                        bg=SECONDARY_COLOR, fg=FG_COLOR, length=150).pack(side="left", padx=5)
            else:
                tk.Label(white_model_frame, text="Variant:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 9)).pack(anchor="w", padx=5)
                for variant in ["baseline", "attack", "est"]:
                    tk.Radiobutton(white_model_frame, text=variant.capitalize(), variable=white_model, value=variant,
                                 bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                                 activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                                 font=("Arial", 8)).pack(anchor="w", padx=15, pady=1)
        
        def update_black_ui(*args):
            """Update Black AI UI."""
            for widget in black_model_frame.winfo_children():
                widget.destroy()
            for widget in black_depth_frame.winfo_children():
                widget.destroy()
            
            if black_ai.get() == "minimax":
                tk.Label(black_depth_frame, text="Depth:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 9)).pack(side="left", padx=5)
                tk.Scale(black_depth_frame, from_=1, to=8, orient="horizontal", variable=black_depth,
                        bg=SECONDARY_COLOR, fg=FG_COLOR, length=150).pack(side="left", padx=5)
            else:
                tk.Label(black_model_frame, text="Variant:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 9)).pack(anchor="w", padx=5)
                for variant in ["baseline", "attack", "est"]:
                    tk.Radiobutton(black_model_frame, text=variant.capitalize(), variable=black_model, value=variant,
                                 bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                                 activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                                 font=("Arial", 8)).pack(anchor="w", padx=15, pady=1)
        
        def on_start():
            if game_mode.get() == "vs_human":
                self.ai_vs_ai_mode = False
                self.AI_type = selected_ai.get()
                self.neural_model_type = selected_model.get()
                self.minimax_depth = selected_depth.get()
                print(f"Human vs AI: Side={self.color}, AI={self.AI_type}, Model/Depth={self.neural_model_type}/{self.minimax_depth}")
            else:
                self.ai_vs_ai_mode = True
                self.white_ai_type = white_ai.get()
                self.white_model_type = white_model.get()
                self.white_minimax_depth = white_depth.get()
                self.black_ai_type = black_ai.get()
                self.black_model_type = black_model.get()
                self.black_minimax_depth = black_depth.get()
                print(f"AI vs AI: White={self.white_ai_type}({self.white_model_type}/{self.white_minimax_depth}), Black={self.black_ai_type}({self.black_model_type}/{self.black_minimax_depth})")
            selection_window.destroy()
        
        # ===== TITLE =====
        title_frame = tk.Frame(selection_window, bg=ACCENT_COLOR, height=50)
        title_frame.pack(fill="x", pady=0)
        tk.Label(title_frame, text="⚔ Chess AI Setup ⚔", font=("Arial", 18, "bold"), 
                bg=ACCENT_COLOR, fg="#000000").pack(pady=10)
        
        # ===== MAIN CONTENT =====
        content_frame = tk.Frame(selection_window, bg=BG_COLOR)
        content_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # ===== GAME MODE SECTION =====
        mode_label = tk.Label(content_frame, text="Game Mode", font=("Arial", 12, "bold"), 
                            bg=BG_COLOR, fg=ACCENT_COLOR)
        mode_label.pack(anchor="w", pady=(0, 10))
        
        mode_buttons_frame = tk.Frame(content_frame, bg=BG_COLOR)
        mode_buttons_frame.pack(fill="x", pady=(0, 20))
        
        for text, value in [("🎮 vs Human", "vs_human"), ("🤖 AI vs AI", "ai_vs_ai")]:
            tk.Radiobutton(mode_buttons_frame, text=text, variable=game_mode, value=value,
                         bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                         activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                         font=("Arial", 10)).pack(anchor="w", pady=5)
        
        # ===== DYNAMIC CONTENT BASED ON MODE =====
        
        # Frame that will hold either "Human vs AI" or "AI vs AI" content
        mode_content = tk.Frame(content_frame, bg=BG_COLOR)
        mode_content.pack(fill="both", expand=True)
        
        # Human vs AI Section
        human_section = tk.Frame(mode_content, bg=BG_COLOR)
        
        tk.Label(human_section, text="Your Side", font=("Arial", 11, "bold"), 
                bg=BG_COLOR, fg=SECONDARY_COLOR).pack(anchor="w", pady=(10, 5))
        side_frame = tk.Frame(human_section, bg=BG_COLOR)
        side_frame.pack(fill="x", pady=(0, 15))
        for text, value in [("⚪ White", "w"), ("⚫ Black", "b")]:
            tk.Radiobutton(side_frame, text=text, variable=tk.StringVar(), value=value,
                         bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                         activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                         command=lambda v=value: setattr(self, 'color', v),
                         font=("Arial", 10)).pack(side="left", padx=10)
        
        tk.Label(human_section, text="Opponent AI Type", font=("Arial", 11, "bold"), 
                bg=BG_COLOR, fg=SECONDARY_COLOR).pack(anchor="w", pady=(10, 5))
        ai_type_frame = tk.Frame(human_section, bg=BG_COLOR)
        ai_type_frame.pack(fill="x", pady=(0, 15))
        for ai_type in ["minimax", "neural_mcts", "neural_direct"]:
            tk.Radiobutton(ai_type_frame, text=ai_type.replace("_", " ").title(), variable=selected_ai, 
                         value=ai_type, bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                         activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                         command=update_human_ui, font=("Arial", 10)).pack(anchor="w", pady=3)
        
        depth_frame = tk.Frame(human_section, bg=BG_COLOR)
        depth_frame.pack(fill="x", pady=(5, 10))
        
        model_frame = tk.Frame(human_section, bg=BG_COLOR)
        model_frame.pack(fill="x", pady=(5, 15))
        
        # AI vs AI Section
        ai_vs_ai_section = tk.Frame(mode_content, bg=BG_COLOR)
        
        # White AI
        tk.Label(ai_vs_ai_section, text="White AI", font=("Arial", 11, "bold"), 
                bg=BG_COLOR, fg=SECONDARY_COLOR).pack(anchor="w", pady=(10, 5))
        white_ai_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        white_ai_frame.pack(fill="x", pady=(0, 8))
        for ai_type in ["minimax", "neural_mcts", "neural_direct"]:
            tk.Radiobutton(white_ai_frame, text=ai_type.replace("_", " ").title(), variable=white_ai,
                         value=ai_type, bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                         activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                         command=update_white_ui, font=("Arial", 9)).pack(anchor="w", pady=2)
        
        white_depth_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        white_depth_frame.pack(fill="x", pady=(3, 10))
        
        white_model_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        white_model_frame.pack(fill="x", pady=(3, 15))
        
        # Black AI
        tk.Label(ai_vs_ai_section, text="Black AI", font=("Arial", 11, "bold"), 
                bg=BG_COLOR, fg=SECONDARY_COLOR).pack(anchor="w", pady=(10, 5))
        black_ai_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        black_ai_frame.pack(fill="x", pady=(0, 8))
        for ai_type in ["minimax", "neural_mcts", "neural_direct"]:
            tk.Radiobutton(black_ai_frame, text=ai_type.replace("_", " ").title(), variable=black_ai,
                         value=ai_type, bg=BG_COLOR, fg=FG_COLOR, selectcolor=ACCENT_COLOR,
                         activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
                         command=update_black_ui, font=("Arial", 9)).pack(anchor="w", pady=2)
        
        black_depth_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        black_depth_frame.pack(fill="x", pady=(3, 10))
        
        black_model_frame = tk.Frame(ai_vs_ai_section, bg=BG_COLOR)
        black_model_frame.pack(fill="x", pady=(3, 15))
        
        def switch_mode(*args):
            human_section.pack_forget()
            ai_vs_ai_section.pack_forget()
            if game_mode.get() == "vs_human":
                human_section.pack(fill="both", expand=True)
                update_human_ui()
            else:
                ai_vs_ai_section.pack(fill="both", expand=True)
                update_white_ui()
                update_black_ui()
        
        game_mode.trace("w", switch_mode)
        
        # Initialize UI
        switch_mode()
        
        # ===== START BUTTON =====
        button_frame = tk.Frame(selection_window, bg=BG_COLOR)
        button_frame.pack(fill="x", padx=20, pady=20)
        
        start_btn = tk.Button(button_frame, text="▶ START GAME", command=on_start,
                            font=("Arial", 12, "bold"), bg=ACCENT_COLOR, fg="#000000",
                            padx=50, pady=12, relief="raised", bd=2, cursor="hand2")
        start_btn.pack(fill="x")
        
        selection_window.wait_window()


# Create an instance and start a game
if __name__ == "__main__":
    # Required for multiprocessing on Windows
    import multiprocessing
    multiprocessing.freeze_support()
    
    game = Main()
    game.start_game()
