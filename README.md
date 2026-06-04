# Chess AI Training System

A comprehensive deep learning system for training a neural network chess AI, featuring both a playable interface and a sophisticated training pipeline with multiple training approaches.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Game Controls](#game-controls)
- [AI Engine](#ai-engine)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Graphical chessboard** with a user-friendly interface for playing games
- **Multi-mode training system** that alternates between professional games, regular games, and self-play
- **Reinforcement learning** through self-play with Monte Carlo Tree Search (MCTS)
- **Tactical pattern recognition** via puzzle training
- **Dynamic batch sizing** to optimize GPU memory usage
- **Mixed precision training** for faster computation
- **Two neural network architectures**:
  - Big model: 10 residual blocks (20 input channels)
  - Small model: 5 residual blocks (18 input channels)

## Dependencies

Before running the chess game, ensure you have the following dependencies installed:

- [Python](https://www.python.org/downloads/): The programming language used for the project.

### Python Libraries

1. Install the required Python libraries using the following command:

   pip install -r requirements.txt

-> chess: A chess library for Python.
-> pygame: A set of Python modules designed for writing video games.
-> numpy: A fundamental package for scientific computing with Python.
-> torch: An open-source machine learning library for neural networks.
-> psutil: For monitoring memory usage during training.

These libraries are specified in the requirements.txt file and will be installed automatically during the setup process.

# Installation
1. Clone the repository:
-> git clone https://github.com/your-username/chess-game.git

2. Navigate to the project directory:
-> cd chess

3. Install the required dependencies:
-> pip install -r requirements.txt

4. (Optional) Add openings pgn files to the folder `/openings`.
-> I used PGN Mentor to find the files.

5. (Optional) Download Stockfish engine and place it in the `/stockfish` folder.

# Usage
1. Run the main.py script to start the chess game:
-> python main.py 

# Training
1. Change directory to the train folder
```bash
cd train
```

2. Download pgn files. I used Lichess and PGN mentor. Place puzzles (both pgn and csv as lichess outputs in csv, but the code handles both) in `/train/chess_pgns/puzzles`. Place high elo games in `/train/chess_pgns` and place professional games in `/train/chess_pgn/pros`.

3. Start training by choosing one of following commands:
- Default usage (will train with all modes, switching mode every few iterations)
```bash
python train.py
```

- Pro game training
```bash
python train.py pro
```

- Self play training
```bash
python train.py self-play [games_per_batch] [iterations_per_cycle]
```

- Regular games training
```bash
python train.py regular
```

- Medium quality and speed self play training
```bash
python train.py self-play --fast-mtcs
```

- Lower quality but faster self play training:
```bash
python train.py self-play --no-mcts
```

- Choose model size:
```bash
python train.py --model small
```

4. Training features:
- **Dynamic batch sizing**: Automatically determines optimal batch size for your GPU
- **Category-based puzzle training**: Weights different tactical patterns differently (mates, forks, pins)
- **Memory management**: Periodically cleans up memory to avoid OOM errors
- **Progressive training parameters**: Weights adjust as training progresses
- **Checkpoint saving**: Save progress between training sessions
- **Tactical recognition testing**: Periodically tests model on tactical positions

# Live Dashboard & Spectate Mode

The training loop exposes a tiny HTTP control plane on `http://127.0.0.1:7860`
that drives a stateless browser dashboard and a Tkinter dashboard, and lets
you watch model-vs-model games and puzzle drills live.

## Running the trainer

```bash
cd train
python train.py
```

The trainer auto-starts the control server on `127.0.0.1:7860` (loopback
only — no LAN exposure). Disable with `LeagueTrainer(enable_control_server=False)`
in your launcher if you don't need it.

## Browser dashboard

Open `http://127.0.0.1:7860/` in any browser. You get:

- **Mode buttons** — `eco` / `balanced` / `boost` performance presets, applied live
- **Auto-mode toggle** — trainer promotes/demotes preset based on CPU usage
- **Pause / resume** — pause training between rounds
- **Resource bars** — CPU, RAM, GPU usage
- **Live charts** — loss (policy/value), throughput (games/min), buffer fill
- **Checkpoint table** — double-click any row to open the spectate modal
- **Spectate modal** — model-vs-model game (configurable visits, start FEN)
  and puzzle drills (consumes `train/cache/puzzles_meta.pkl`)

All state lives in the trainer — the dashboard is a stateless consumer.

## Tkinter dashboard (separate process)

```bash
cd train
python -m league.dashboard_tk
```

Same features as the browser but as a native window, polls `/api/status`
every 2 seconds. Right-click context menus and double-click handlers mirror
the browser behaviour. Useful when you want a dedicated monitor window
without keeping a browser tab open.

## HTTP API quick reference

```bash
# Status snapshot
curl -s http://127.0.0.1:7860/api/status | python -m json.tool

# Switch mode
curl -X POST http://127.0.0.1:7860/api/mode -d '{"mode":"boost"}'

# Hot-swap a knob
curl -X POST http://127.0.0.1:7860/api/knobs -d '{"knobs":{"BATCH_SIZE":512}}'

# Pause / resume
curl -X POST http://127.0.0.1:7860/api/pause -d '{"paused":true}'

# Watch live match events (Server-Sent Events)
curl -N http://127.0.0.1:7860/api/matches/stream
```

Full schema is in `train/league/control_server.py` (search for
`URL_PATTERNS`).

## Performance presets

| Preset     | BATCH | Workers | MCTS visits | Replay buf | Use case                |
|------------|------:|--------:|------------:|-----------:|-------------------------|
| `eco`      |   128 |       3 |          80 |      50K   | Light load (working PC) |
| `balanced` |   256 |       6 |         200 |     100K   | Default                 |
| `boost`    |  1024 |      12 |         400 |     300K   | Overnight / idle GPU    |

Switch at runtime with `set_mode("boost")` or via the dashboard. The
trainer persists the active mode across restarts (`train/perf_mode.json`).

## Spectate mode

Queue a model-vs-model game or a puzzle drill from the dashboard or the
HTTP API:

```bash
# Model vs model
curl -X POST http://127.0.0.1:7860/api/matches -d '{
  "type": "model",
  "params": {"white": "baseline", "black": "attack", "visits": 200}
}'

# Puzzle drill (random from sidecar)
curl -X POST http://127.0.0.1:7860/api/matches -d '{
  "type": "puzzle",
  "params": {"visits": 100}
}'
```

Events stream over `/api/matches/stream` (SSE) in this order:
`start → move / drill_move → done` (or `error`).

To use a specific checkpoint in spectate, use `<variant>_step_<N>` as the
model name, e.g. `baseline_step_35`. The trainer resolves it to a
`.pt` file under `train/checkpoints/`.

### Puzzle sidecar

The puzzle cache (`train/cache/puzzle_tensors/*.pkl`) does **not** store
FENs or solution lines — it only has the input tensors. To run puzzle
drills you need a small sidecar that maps `puzzle_id → {fen, solution, ...}`:

```bash
# Build the sidecar (one-time, ~10s for 2.4M puzzles)
cd train
python -m league.puzzle_sidecar
# or, from the repo root:
python train/build_puzzle_sidecar.py
```

This writes `train/cache/puzzles_meta.pkl` (~300MB for the full Lichess
DB). The spectate worker loads it lazily on the first drill request and
keeps it in memory. Re-run the command if you update your puzzle CSV.

## Hot-swap knobs

Most training knobs are changeable at runtime via `set_knob()` or the
`/api/knobs` endpoint. Changes are batched and applied at the next safe
checkpoint (round boundary for structural knobs, training step for
hyperparameters). See `train/TUNING_REFERENCE.md` for the full list.

# Game Controls
-> Click on a piece to select it.
-> Drag the selected piece to the desired square to make a move.
-> Release the mouse button to complete the move.

# AI Engine
The AI engine features two approaches:

1. **Neural Network Models**:
   - **Alpha Zero baseline**
   - **Alpha Zero with Attack Maps**
   - **Custom Model**: Separate policy and value sooner, keeping the shared trunk short to reduce gradient interference and improve specialization.
      - **Arquitecture**:

      ```
            Input
            ↓
            Shared Trunk (5 residual blocks)
            ↓
      ┌───────────────┐
      │               │
      Policy Trunk     Value Trunk
      (5 blocks)       (5 blocks)
      │               │
      Policy Head      Value Head
      ```

2. **Minimax**


You can choose between these AI modes in the game settings or training parameters.

# Contributing
Contributions are welcome! Feel free to open issues or pull requests for any improvements or new features.

# License
This project is licensed under the MIT License - see the LICENSE file for details.
