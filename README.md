# Chess Game with AI

This is a simple chess game implemented in Python using the Pygame library. The game includes a basic graphical user interface and allows the player to play against an AI opponent. The project also features two AI models built with Pytorch. One of them smaller and the other one bigger.

## Table of Contents

- [Features](#features)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Game Controls](#game-controls)
- [AI Engine](#ai-engine)
- [Contributing](#contributing)
- [License](#license)

## Features

- Graphical chessboard with a user-friendly interface.
- Ability to play against an AI opponent.
- AI engine with basic evaluation function and minimax algorithm with alpha-beta pruning.
- Two neural AI models created with PyTorch and TensorFlow.

## Dependencies

Before running the chess game, ensure you have the following dependencies installed:

- [Python](https://www.python.org/downloads/): The programming language used for the project.

### Python Libraries

1. Install the required Python libraries using the following command:

   pip install -r requirements.txt

-> chess: A chess library for Python.

-> pygame: A set of Python modules designed for writing video games.

-> numpy: A fundamental package for scientific computing with Python.

-> torch: An open-source machine learning library used for the other AI model.



These libraries are specified in the requirements.txt file and will be installed automatically during the setup process.

# Instalation
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
- Default usage (will train with all modes, switching mode every 5 iterations)
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

- Lower quality but faster self play training:
```bash
python train.py self-play --no-mcts
```

> **NOTE:** All training modes will use the puzzles in the training and do a simple tactical training. 

# Note
There are different code snippets in the start_game_function and in the play_engine_move in the main.py file, allowing you to play against various AI models or observe AI vs AI matches.

# Game Controls
-> Click on a piece to select it.

-> Drag the selected piece to the desired square to make a move.

-> Release the mouse button to complete the move.

# AI Engine
The AI engine uses a basic evaluation function and the minimax algorithm with alpha-beta pruning to make decisions. The depth of the search tree is configurable in the main.py file.

Additionally, two neural AI models are included:

-> PyTorch Model: A neural network model trained with PyTorch for enhanced decision-making.
You can choose between these AI models in the game settings.

# Contributing
Contributions are welcome! Feel free to open issues or pull requests for any improvements or new features.

# License
This project is licensed under the MIT License - see the LICENSE file for details.