import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import glob
import json
import signal
import sys
import itertools
import csv
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

# Chess Neural Network with adjustable blocks and channels
class ChessNet(nn.Module):
    def __init__(self, num_blocks=10, channels=256):  # Increased to 10 blocks
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(20, channels, kernel_size=3, padding=1)  # 20 channels for added features
        self.bn1 = nn.BatchNorm2d(channels)
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])
        self.policy_conv = nn.Conv2d(channels, 73, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(73)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.blocks:
            x = block(x)
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 73 * 8 * 8)
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 64)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        return policy, value

# Updated board_to_tensor with repetition counter and move number
def board_to_tensor(board, move_number):
    tensor = np.zeros((20, 8, 8), dtype=np.float32)  # Increased to 20 channels
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            for square in board.pieces(piece_type, color):
                row, col = divmod(square, 8)
                channel = piece_type - 1 if color == chess.WHITE else piece_type + 5
                tensor[channel, row, col] = 1
    tensor[12, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    tensor[13, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    tensor[14, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    tensor[15, :, :] = board.has_queenside_castling_rights(chess.BLACK)
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[16, row, col] = 1
    tensor[17, :, :] = 1 if board.turn == chess.WHITE else 0
    tensor[18, :, :] = board.halfmove_clock / 50.0  # Normalized repetition counter (fifty-move rule)
    tensor[19, :, :] = move_number / 200.0  # Normalized move number (assuming max 200 moves)
    return tensor

# Move Index Mapping (unchanged)
promotion_moves = {}
promotion_idx = 4096
for rank in [6, 1]:
    for col in range(8):
        from_square = chess.square(col, rank)
        to_square = chess.square(col, rank + (1 if rank == 6 else -1))
        for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            promotion_moves[(from_square, to_square, piece)] = promotion_idx
            promotion_idx += 1
        if col > 0:
            to_square = chess.square(col - 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1
        if col < 7:
            to_square = chess.square(col + 1, rank + (1 if rank == 6 else -1))
            for piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                promotion_moves[(from_square, to_square, piece)] = promotion_idx
                promotion_idx += 1

def get_move_index(move):
    if move.promotion:
        return promotion_moves[(move.from_square, move.to_square, move.promotion)]
    return move.from_square * 64 + move.to_square

# Chess Dataset with Symmetry Augmentation
class ChessDataset(Dataset):
    def __init__(self, games, augment=True):
        self.positions = []
        self.augment = augment
        for game in games:
            result_str = game.headers.get("Result", "*")
            if result_str not in ["1-0", "0-1", "1/2-1/2"]:
                continue
            result = {'1-0': 1, '0-1': -1, '1/2-1/2': 0}[result_str]
            board = game.board()
            move_number = 1
            for move in game.mainline_moves():
                input_tensor = board_to_tensor(board, move_number)
                policy_target = get_move_index(move)
                value_target = result if board.turn == chess.WHITE else -result
                self.positions.append((input_tensor, policy_target, value_target))
                if self.augment:  # Mirror horizontally
                    mirrored_board = board.mirror()
                    mirrored_move = chess.Move(
                        chess.square_mirror(move.from_square),
                        chess.square_mirror(move.to_square),
                        move.promotion
                    )
                    mirrored_tensor = board_to_tensor(mirrored_board, move_number)
                    mirrored_policy = get_move_index(mirrored_move)
                    self.positions.append((mirrored_tensor, mirrored_policy, value_target))
                board.push(move)
                move_number += 1

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        input_tensor, policy_target, value_target = self.positions[idx]
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

# Puzzle Dataset (unchanged for now)
class PuzzleDataset(Dataset):
    def __init__(self, puzzles):
        self.puzzles = puzzles

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, idx):
        fen, move_uci, value_target = self.puzzles[idx]
        board = chess.Board(fen)
        move = chess.Move.from_uci(move_uci)
        input_tensor = board_to_tensor(board, 0)  # Move number not tracked in puzzles
        policy_target = get_move_index(move)
        return (torch.tensor(input_tensor, dtype=torch.float32),
                torch.tensor(policy_target, dtype=torch.long),
                torch.tensor(value_target, dtype=torch.float32))

# Load puzzles (unchanged)
def load_puzzles(pgn_file):
    puzzles = []
    with open(pgn_file, encoding='ISO-8859-1') as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            board = game.board()
            try:
                best_move = list(game.mainline_moves())[0]
                fen = board.fen()
                move_uci = best_move.uci()
                puzzles.append((fen, move_uci, 1.0))
            except IndexError:
                continue
    return puzzles

def load_lichess_puzzles(csv_file):
    puzzles = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row['FEN']
            moves = row['Moves'].split()
            if moves:
                move_uci = moves[0]
                value_target = 1.0 if 'mate' in row['Themes'].lower() else 0.5
                puzzles.append((fen, move_uci, value_target))
    return puzzles

# Load games in batches (unchanged)
def load_games_in_batches(pgn_files, state_file, batch_size=1500):
    total_games = 0
    processed_games = 0
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            processed_games = state.get("processed_games", 0)
        print(f"Resuming from {processed_games} processed games")
    games = []
    for file in pgn_files:
        with open(file) as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break
                total_games += 1
                if total_games <= processed_games:
                    continue
                games.append(game)
                if len(games) == batch_size:
                    yield games
                    processed_games += len(games)
                    games = []
    if games:
        yield games
        processed_games += len(games)
    with open(state_file, 'w') as f:
        json.dump({"processed_games": processed_games}, f)

# Self-play game generation (basic implementation)
def generate_self_play_games(model, num_games=100):
    games = []
    model.eval()
    for _ in range(num_games):
        board = chess.Board()
        game = chess.pgn.Game()
        game.headers["Result"] = "*"
        node = game
        move_number = 1
        while not board.is_game_over():
            input_tensor = torch.tensor(board_to_tensor(board, move_number), dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, _ = model(input_tensor)
            policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            legal_moves = list(board.legal_moves)
            move_probs = np.zeros(len(legal_moves))
            for idx, move in enumerate(legal_moves):
                move_idx = get_move_index(move)
                move_probs[idx] = policy[move_idx]
            move_probs /= move_probs.sum()  # Normalize
            move = np.random.choice(legal_moves, p=move_probs)
            board.push(move)
            node = node.add_variation(move)
            move_number += 1
        result = board.result()
        game.headers["Result"] = result
        games.append(game)
    return games

# Updated Training Function with Weighted Loss and Scheduler
def train_batch(model, game_dataset, puzzle_dataset, save_path, state_file, epochs=5, processed_games=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added L2 regularization
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)  # Cosine annealing scheduler
    policy_loss_fn = nn.CrossEntropyLoss()
    value_loss_fn = nn.MSELoss()

    game_dataloader = DataLoader(game_dataset, batch_size=32, shuffle=True)
    puzzle_dataloader = DataLoader(puzzle_dataset, batch_size=32, shuffle=True)
    puzzle_iter = itertools.cycle(puzzle_dataloader)

    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state = json.load(f)
            start_epoch = state.get("last_epoch", 0)
            print(f"Resuming training from epoch {start_epoch + 1}")
    else:
        state = {"processed_games": 0, "last_epoch": 0}
        start_epoch = 0

    puzzle_frequency = 1
    policy_weight = 1.5  # Emphasize policy early
    value_weight = 1.0

    for epoch in range(start_epoch, epochs + start_epoch):
        model.train()
        total_loss = 0
        game_batch_count = 0
        for game_batch in game_dataloader:
            inputs, policy_targets, value_targets = game_batch
            inputs = inputs.to(device)
            policy_targets = policy_targets.to(device)
            value_targets = value_targets.to(device)
            optimizer.zero_grad()
            policy_logits, value_pred = model(inputs)
            policy_loss = policy_loss_fn(policy_logits, policy_targets)
            value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
            loss = policy_weight * policy_loss + value_weight * value_loss  # Weighted loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            game_batch_count += 1

            if game_batch_count % puzzle_frequency == 0:
                puzzle_batch = next(puzzle_iter)
                inputs, policy_targets, value_targets = puzzle_batch
                inputs = inputs.to(device)
                policy_targets = policy_targets.to(device)
                value_targets = value_targets.to(device)
                optimizer.zero_grad()
                policy_logits, value_pred = model(inputs)
                policy_loss = policy_loss_fn(policy_logits, policy_targets)
                value_loss = value_loss_fn(value_pred.squeeze(), value_targets)
                loss = policy_weight * policy_loss + value_weight * value_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        scheduler.step()  # Update learning rate
        num_puzzle_batches = len(game_dataloader) // puzzle_frequency
        avg_loss = total_loss / (len(game_dataloader) + num_puzzle_batches)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        state["last_epoch"] = epoch + 1
        state["processed_games"] = processed_games + len(game_dataset)
        torch.save(model.state_dict(), save_path)
        with open(state_file, 'w') as f:
            json.dump(state, f)
        print(f"Checkpoint saved at epoch {epoch + 1}")

# Signal handler (updated)
def signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch):
    print("\nTraining interrupted! Saving model and state...")
    torch.save(model.state_dict(), save_path)
    with open(state_file, 'w') as f:
        state = {"processed_games": processed_games, "last_epoch": current_epoch + 1}
        json.dump(state, f)
    print(f"Model saved to {save_path}, state saved to {state_file}")
    sys.exit(0)



# Initialize model
model = ChessNet(num_blocks=10, channels=256).to(device)  # Adjustable blocks and channels
save_path = "./chess_model/chess_model.pth"
state_file = "./chess_model/training_state.json"

if os.path.exists(save_path):
    model.load_state_dict(torch.load(save_path))
    print(f"Loaded existing model from {save_path}")

current_epoch = 0
processed_games = 0
if os.path.exists(state_file):
    with open(state_file, 'r') as f:
        state = json.load(f)
        processed_games = state.get("processed_games", 0)
        current_epoch = state.get("last_epoch", 0)

import gc


# Main execution with self-play integration
if __name__ == "__main__":
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))
    signal.signal(signal.SIGTERM, lambda sig, frame: signal_handler(sig, frame, model, save_path, state_file, processed_games, current_epoch))

    pgn_directory = "./chess_pgns"
    pgn_files = glob.glob(os.path.join(pgn_directory, "*.pgn"))

    puzzle_pgn = "./chess_pgns/puzzles/puzzles.pgn"
    pgn_puzzles = load_puzzles(puzzle_pgn)
    print(f"Loaded {len(pgn_puzzles)} PGN puzzles")

    lichess_csv = "./chess_pgns/puzzles/lichess_db_puzzle.csv"
    lichess_puzzles = load_lichess_puzzles(lichess_csv)
    print(f"Loaded {len(lichess_puzzles)} Lichess puzzles")

    all_puzzles = pgn_puzzles + lichess_puzzles
    puzzle_dataset = PuzzleDataset(all_puzzles)
    print(f"Total puzzles: {len(all_puzzles)}")


    # Process games in batches and integrate self-play
    for batch_num, games in enumerate(load_games_in_batches(pgn_files, state_file, batch_size=1500), 1):
        if not games:
            continue
        print(f"\nProcessing batch {batch_num} with {len(games)} games")
        game_dataset = ChessDataset(games, augment=True)
        train_batch(model, game_dataset, puzzle_dataset, save_path, state_file, epochs=5, processed_games=processed_games)
        processed_games += len(games)
        del game_dataset
        gc.collect()

        # Generate and train on self-play games every batch
        print("Generating self-play games...")
        self_play_games = generate_self_play_games(model, num_games=1)  # Small number to start
        self_play_dataset = ChessDataset(self_play_games, augment=True)
        train_batch(model, self_play_dataset, puzzle_dataset, save_path, state_file, epochs=2, processed_games=processed_games)
        del self_play_dataset
        gc.collect()

    print("Training completed!")
