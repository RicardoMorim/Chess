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

# Game Controls
-> Click on a piece to select it.
-> Drag the selected piece to the desired square to make a move.
-> Release the mouse button to complete the move.

# AI Engine
The AI engine features two approaches:

1. **Neural Network Models**:
   - **Big Model**: 20 input channels, 10 residual blocks with dual policy/value head
   - **Small Model**: 18 input channels, 5 residual blocks with dual policy/value head

2. **Monte Carlo Tree Search (MCTS)**:
   - Policy-guided tree search with UCB formula
   - Parallel MCTS for faster move selection
   - Dirichlet noise at root for exploration (AlphaZero style)
   - Tree reuse between moves

3. **Tactical Recognition**:
   - Specialized training on chess puzzles
   - Category-based learning (checkmates, forks, pins)
   - Periodic verification of tactical ability

You can choose between these AI modes in the game settings or training parameters.

# Contributing
Contributions are welcome! Feel free to open issues or pull requests for any improvements or new features.

# License
This project is licensed under the MIT License - see the LICENSE file for details.