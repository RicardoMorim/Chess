# Chess AI Training System

## Architecture

```
train/
├── core/                    # Core components (shared across all training modes)
│   ├── __init__.py          # Re-exports: models, mcts, training, data
│   └── (models.py, mcts.py, training.py, data.py are in parent)
│
├── bootstrap/               # One-time cold-start training (NOT in main loop)
│   ├── __init__.py
│   ├── puzzle_train.py      # Supervised learning on puzzles
│   ├── puzzle_eval.py       # Puzzle accuracy evaluation
│   └── stockfish_filter.py  # Puzzle quality validation
│
├── league/                  # Main training system (MCTS self-play)
│   ├── __init__.py
│   ├── main.py              # Entry point: python league/main.py
│   ├── league_trainer.py    # Orchestrator
│   ├── replay_buffer.py     # Per-model FIFO buffer
│   ├── self_play_worker.py  # Parallel MCTS workers
│   ├── evaluator.py         # Regression detection
│   └── monitoring.py        # Metrics collection
│
├── tools/                   # Utility scripts
│   └── evaluate_with_stockfish.py
│
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs and metrics
└── docs/                    # Documentation
```

## THE RULE

> **Only self-play with MCTS improves models long-term.**
> Everything else exists only to:
> 1. Reduce cold start (bootstrap)
> 2. Catch regressions (evaluation)
> 3. Measure strength (monitoring)

## Quick Start

### 1. Bootstrap (Optional, One-Time)

```bash
cd train
python bootstrap/puzzle_train.py --puzzle-file chess_pgns/puzzles/puzzles.csv --epochs 5 --variant baseline
```

### 2. Run League Training

```bash
cd train
python league/main.py
```

### 3. Monitor

```bash
tail -f logs/metrics.log
```

### 4. Evaluate

```bash
python bootstrap/puzzle_eval.py --checkpoint checkpoints/baseline_step_50.pt --puzzle-file puzzles.csv
```

## Components

### Core (`core/`)

Re-exports from parent directory:
- `create_model(variant)` - Create model instances
- `MCTS` - Monte Carlo Tree Search
- `board_to_tensor()` - Board representation
- `PolicyLoss`, `ValueLoss` - Loss functions

```python
from core import create_model, MCTS, board_to_tensor
```

### Bootstrap (`bootstrap/`)

One-time supervised learning to reduce cold-start:

- **puzzle_train.py**: Train on tactical puzzles
- **puzzle_eval.py**: Measure puzzle accuracy
- **stockfish_filter.py**: Validate puzzle quality

**Important**: Bootstrap is NOT part of the main training loop. Run it once, then use league training.

### League (`league/`)

Main training system with parallel self-play:

- **main.py**: Entry point
- **league_trainer.py**: Orchestrates rounds
- **replay_buffer.py**: Stores game data
- **self_play_worker.py**: CPU workers for MCTS games
- **evaluator.py**: Low-frequency regression tests
- **monitoring.py**: Centralized metrics

```python
from league import LeagueTrainer
trainer = LeagueTrainer()
trainer.run()
```

## Training Loop

Each round:

1. **Self-Play (CPU parallel)**: 6 workers × 10 games = 60 games
2. **Training (GPU batched)**: 50 steps × 256 batch = 12,800 positions
3. **Checkpointing**: Every 5 rounds
4. **Evaluation**: Every 10 rounds vs old checkpoints
5. **Metrics**: Logged every round

## Configuration

Edit `league/league_trainer.py`:

```python
class LeagueTrainer:
    VARIANTS = ["baseline", "attack", "est"]
    NUM_SELF_PLAY_WORKERS = 6
    GAMES_PER_WORKER_PER_ROUND = 10
    BATCH_SIZE = 256
    TRAINING_STEPS_PER_ROUND = 50
    MCTS_VISITS_TRAINING = 800
```

## Monitoring

Metrics are automatically collected:

- **losses**: policy_loss, value_loss, total_loss
- **games**: game_length, outcomes
- **buffer**: size, fill_ratio
- **evaluation**: win_rate, elo_diff

View in:
- `logs/metrics.log` (real-time)
- `logs/metrics_round_N.json` (periodic snapshots)

## Model Variants

| Variant | Input Channels | Description |
|---------|---------------|-------------|
| baseline | 18 | Basic piece + game state |
| attack | 22 | + attack maps |
| est | 22 | Early Split Trunk architecture |

## Troubleshooting

### Import errors

```bash
cd train
python -c "from core import create_model; print('OK')"
```

### GPU out of memory

Reduce `BATCH_SIZE` from 256 to 128.

### Workers crash

Check if MCTS works:
```bash
python -c "from core import MCTS, create_model; print('OK')"
```

## Next Steps

1. Run `python league/main.py`
2. Monitor with `tail -f logs/metrics.log`
3. Check checkpoints in `checkpoints/`
4. Evaluate with `bootstrap/puzzle_eval.py`
