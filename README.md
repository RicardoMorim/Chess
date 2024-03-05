# Chess Game with AI

This is a simple chess game implemented in Python using the Pygame library. The game includes a basic graphical user interface and allows the player to play against an AI opponent.

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

## Dependencies

Before running the chess game, ensure you have the following dependencies installed:

- [Python](https://www.python.org/downloads/): The programming language used for the project.

### Python Libraries

1. Install the required Python libraries using the following command:

   pip install -r requirements.txt

-> chess: A chess library for Python.

-> pygame: A set of Python modules designed for writing video games.

These libraries are specified in the requirements.txt file and will be installed automatically during the setup process.

## Installation

1. Clone the repository:

   git clone https://github.com/your-username/chess-game.git

2. Navigate to the project directory:

   cd chess

3. Install the required dependencies:

   pip install -r requirements.txt

## Usage

1. Run the main.py script to start the chess game:

   python main.py

## Game Controls

-> Click on a piece to select it.
-> Drag the selected piece to the desired square to make a move.
-> Release the mouse button to complete the move.

## AI Engine

The AI engine uses a basic evaluation function and the minimax algorithm with alpha-beta pruning to make decisions.
The depth of the search tree is configurable in the ChessEngine.py file.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests for any improvements or new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
