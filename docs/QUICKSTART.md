# League Training: Quick Start Guide

## Installation

1. Ensure you have the required dependencies in `requirements.txt`:

```bash
pip install torch chess numpy
```

2. The league training system is in `league/` directory:

```
chess/
├── league/              # Main training system
│   ├── league_trainer.py
│   ├── replay_buffer.py
│   ├── self_play_worker.py
│   ├── evaluator.py
│   └── monitoring.py
├── train/               # Existing code (models, MCTS, data)
└── checkpoints/         # Where models will be saved
```

## Basic Usage

### Option A: Quick Start (Copy & Run)

Create `league/main.py`:

```python
#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '/'.join(__file__.split('/')[:-2]))

from league.league_trainer import LeagueTrainer
from train.models import create_model

if __name__ == "__main__":
    # Initialize trainer
    trainer = LeagueTrainer(
        checkpoint_dir="checkpoints",
        log_dir="logs",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Initialize models
    trainer.initialize_models(
        model_constructor=create_model,
        model_configs={
            "baseline": {"input_channels": 18, "num_blocks": 15},
            "attack": {"input_channels": 22, "num_blocks": 15},
            "est": {"input_channels": 22, "num_blocks": 20},
        }
    )
    
    # Run training (will run indefinitely until interrupted)
    trainer.run(max_rounds=None)
```

Run it:

```bash
cd chess
python league/main.py
```

### Option B: Custom Integration

```python
from league.league_trainer import LeagueTrainer
from my_models import MyChessNet

trainer = LeagueTrainer(device="cuda")
trainer.initialize_models(
    model_constructor=MyChessNet,
    model_configs={...}
)

# Run N rounds
trainer.run(max_rounds=100)

# Or run step by step
for round in range(100):
    for variant in trainer.VARIANTS:
        trainer.generate_self_play(variant)
        trainer.train_model(variant)
    trainer.metrics.log_summary()
```

## What Happens Each Round

1. **Self-Play (2-3 min)**
   - 6 CPU workers run MCTS in parallel
   - Each plays 10 games
   - Data queued to replay buffer

2. **Training (30-60 sec)**
   - 50 SGD steps on 256-batch samples
   - Policy + value losses computed
   - Model weights updated

3. **Metrics (automated)**
   - Buffer fill ratio, game lengths, losses logged
   - Saved to `logs/metrics_round_N.json`

4. **Checkpoint (every 5 rounds)**
   - Model state saved to `checkpoints/variant_step_N.pt`
   - Used for regression detection

5. **Evaluation (every 10 rounds)**
   - Current model tested vs old checkpoint
   - ELO difference computed
   - Regression alert if significant drop

## Monitoring Training

### Real-Time Logs

```bash
tail -f logs/metrics.log
```

### Recent Metrics

```bash
cat logs/metrics_round_50.json | jq '.variants'
```

### All Checkpoints

```bash
ls -lh checkpoints/
```

## Key Files & What They Do

| File | Purpose | When to Modify |
|------|---------|----------------|
| `league_trainer.py` | Main loop orchestrator | Tuning round structure |
| `replay_buffer.py` | Stores game data | Changing buffer size |
| `self_play_worker.py` | Generates games via MCTS | Modifying MCTS parameters |
| `monitoring.py` | Metrics collection | Adding custom metrics |
| `evaluator.py` | Tests vs checkpoints | Changing evaluation rules |

## Configuration

Key hyperparameters in `LeagueTrainer`:

```python
VARIANTS = ["baseline", "attack", "est"]           # Model variants
NUM_SELF_PLAY_WORKERS = 6                          # CPU workers
GAMES_PER_WORKER_PER_ROUND = 10                    # Games per worker
BATCH_SIZE = 256                                   # Training batch
TRAINING_STEPS_PER_ROUND = 50                      # SGD steps per round
CHECKPOINT_EVERY_N_ROUNDS = 5                      # Save freq
EVAL_EVERY_N_ROUNDS = 10                           # Eval freq

MCTS_VISITS_TRAINING = 800                         # MCTS simulations
TEMPERATURE = 1.0                                  # Move randomness
C_PUCT = 4.0                                       # MCTS exploration
```

To change, modify `LeagueTrainer` class or pass as arguments.

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'mcts'"

Make sure you're in the `chess/` directory and have the right Python path.

```bash
cd /path/to/chess
python -c "from train.mcts import MCTS; print('OK')"
```

### ❌ "Workers timeout / stuck"

Check if model inference works:

```bash
python -c "
from train.models import create_model
from train.data import board_to_tensor
import torch
import chess

model = create_model().cuda()
board = chess.Board()
x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).cuda()
with torch.no_grad():
    p, v = model(x)
print('Model OK:', p.shape, v.shape)
"
```

### ❌ "CUDA out of memory"

Reduce `BATCH_SIZE` or `NUM_SELF_PLAY_WORKERS`:

```python
trainer.BATCH_SIZE = 128
trainer.NUM_SELF_PLAY_WORKERS = 4
```

### ❌ "Training loss stuck / not decreasing"

Check:
1. Learning rate: Try `trainer.optimizers[variant].param_groups[0]['lr'] = 0.001`
2. Buffer quality: Ensure MCTS is finding good moves
3. Model size: Try `num_blocks=20` instead of `15`

## Example: Monitor Progress

```python
import json
from pathlib import Path

logs_dir = Path("logs")
for metrics_file in sorted(logs_dir.glob("metrics_round_*.json")):
    with open(metrics_file) as f:
        data = json.load(f)
    
    round_num = metrics_file.stem.split("_")[-1]
    print(f"Round {round_num}:")
    for variant, stats in data["variants"].items():
        print(f"  {variant}: loss={stats.get('recent_loss', 0):.3f}, "
              f"buffer={stats.get('buffer_fill_ratio', 0):.1%}")
```

## Next Steps

1. **Start training**: Run `python league/main.py`
2. **Monitor**: Watch `logs/metrics.log` and checkpoint directory
3. **Tune**: Adjust hyperparameters based on GPU/CPU utilization
4. **Evaluate**: Check evaluation results after 20-30 rounds
5. **Iterate**: Continue until models converge

---

**The Rule**: Only MCTS self-play improves models. Everything else is measurement, cold-start reduction, or regression detection.
