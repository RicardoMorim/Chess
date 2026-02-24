# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## High‑level Architecture

The project is a **chess AI system** that combines a **playable graphical interface** (Python + pygame) with a **training pipeline** for neural network models and a fast **Minimax engine**.  The code is split into three main logical areas:

1. **Gameplay** – `main.py` and the surrounding utility modules.  The GUI renders a board and lets a human play against an AI.  AI selection is done through the settings dialog and can be either the neural network or the Minimax engine.
2. **Training** – located under the `train/` folder.  It contains a `train.py` driver, a set of utility modules in `train/core`, and a `chess_pgns/` data folder for human games, professional games, and puzzles.  The training system supports several modes:
   * **Self‑play** – MCTS + policy/value network.
   * **Professional / regular games** – policy network trained on curated PGNs.
   * **Puzzle training** – tactical pattern recognition.
   * **Mixed‑mode** – automatic switching between modes.
3. **Engine** – Cython‑accelerated Minimax implementation (`minimax_cy.pyx` → `minimax_cy.c` → compiled extension).  The Python wrapper (`Minimax.py`) exposes a `MinimaxAI` class that can run either single‑threaded or parallel (Lazy‑SMP) search.

### Key files

- `train/core/models.py` – neural‑network definitions (big/small).  The models are PyTorch `nn.Module` subclasses.
- `train/core/training.py` – high‑level training loop, checkpointing, dynamic batch sizing.
- `train/core/mcts.py` – Monte‑Carlo Tree Search implementation.
- `train/tools/evaluate_with_stockfish.py` – benchmarking script that runs Stockfish to validate model moves.
- `Minimax.py` / `Minimax_improved.py` – Python wrappers around the compiled Cython engine.
- `setup.py` and `compile_cython.py` – build the Cython extension.
- `Main.py` – entry point for the GUI.

## Common Development Tasks

| Task | Command | Notes |
|------|---------|-------|
| **Install dependencies** | `pip install -r requirements.txt` | Runs in a virtual environment (recommended). |
| **Build Cython extension** | `python compile_cython.py` | Generates `minimax_cy.cp313-win_amd64.pyd` under `build/`. |
| **Run the game** | `python main.py` | Starts the pygame UI. |
| **Start training (default, mixed mode)** | `python train/train.py` | Uses all available PGN data and switches modes automatically. |
| **Start training (professional games only)** | `python train/train.py pro` |
| **Start training (self‑play)** | `python train/train.py self-play [games_per_batch] [iterations_per_cycle]` |
| **Start training (regular games only)** | `python train/train.py regular` |
| **Start training (fast self‑play)** | `python train/train.py self-play --fast-mtcs` |
| **Start training (no MCTS)** | `python train/train.py self-play --no-mcts` |
| **Select model size** | Append `--model small` or `--model big` to any of the above commands |
| **Run tests** | `python -m unittest discover` or `python -m pytest` (if pytest installed) |
| **Run single test** | `python -m unittest test_parallel.py` |
| **Run Minimax test** | `python test_parallel.py` |

## Advanced Training Optimizations

As of the latest session, the training pipeline includes three major performance improvements:

1. **Disk Usage Guardrails** – Automatically prunes old replay buffer files when disk space is low, keeping only the 3 most recent buffers per variant. Checks every 10 rounds.

2. **Adaptive MCTS Visitation Tuning** – Monitors games/min from recent self-play and adjusts MCTS visits (between 6–32) to maintain a target throughput of ~10 games/min. Adjusts every 5 rounds.

3. **GPU-Batched Inference (Framework)** – Optional infrastructure for aggregating board evaluations from multiple CPU workers into batches for efficient GPU forward passes. Currently a framework; requires `_board_to_features()` and `_move_to_index()` implementations in `gpu_inference_server.py` for use.

For detailed tuning and usage, see [train/OPTIMIZATIONS.md](train/OPTIMIZATIONS.md).

## Running Tests

The project ships with a small test suite under `test_parallel.py`.  It compares single‑threaded and parallel Minimax search.  To execute:

```bash
python -m unittest test_parallel.py
```

or simply:

```bash
python test_parallel.py
```

If you add new tests, place them in a `tests/` directory and run `python -m unittest discover`.

## Training Data Organization

Place your PGN files under the following directories inside `train/chess_pgns/`:

- `high_elo/` – High‑rating games (e.g., Lichess‑rated).
- `pros/` – Professional or classical games.
- `puzzles/` – Tactical puzzles (PGN or CSV).

The training script automatically scans these folders.  Adding a new folder will cause the data loader to include it in the next training run.

## Configuring GPU / CUDA

Training relies on PyTorch.  For GPU acceleration, install a CUDA‑enabled PyTorch wheel that matches your system.  The simplest way is:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Replace `cu118` with the CUDA version you have.

## Quick Links

- **Main code** – `Main.py`
- **Training entry** – `train/train.py`
- **Minimax engine** – `Minimax.py` / `Minimax_improved.py`
- **Cython build** – `compile_cython.py`
- **Model definitions** – `train/core/models.py`
- **Documentation** – `README.md`

Feel free to open issues or pull requests to improve this guide.
