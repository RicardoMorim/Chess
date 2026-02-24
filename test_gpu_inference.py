"""
Quick test for GPU inference server with board encoding implementations.
"""
import torch
import chess
from train.league.gpu_inference_server import GPUInferenceServer
from train.core.models import create_model

# Create a dummy model (small size for fast testing)
model = create_model(variant="baseline", num_blocks=4, channels=64)
model.eval()

# Create server (use CPU for testing, no GPU required)
server = GPUInferenceServer(model, device="cpu")

# Test _board_to_features
board = chess.Board()
features = server._board_to_features(board)
print(f"✓ _board_to_features works: shape={features.shape}, dtype={features.dtype}")

# Test _move_to_index
move = list(board.legal_moves)[0]
idx = server._move_to_index(move)
print(f"✓ _move_to_index works: move={move}, index={idx}")

# Test evaluate_batch
boards = [chess.Board() for _ in range(3)]
results = server.evaluate_batch(boards)
print(f"✓ evaluate_batch works: {len(results)} results")

print("\n✅ All GPU inference tests passed!")
