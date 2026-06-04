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

## Live monitoring (Fase 2/3)

The trainer auto-starts an HTTP control server on `http://127.0.0.1:7860`
when constructed. The browser dashboard lives at `/`; the Tkinter
dashboard runs as a separate process.

```bash
# Open the browser dashboard
xdg-open http://127.0.0.1:7860/    # Linux
start http://127.0.0.1:7860/       # Windows

# Or run the Tkinter dashboard
cd train && python -m league.dashboard_tk
```

You can:
- Switch performance mode (`eco` / `balanced` / `boost`) with a click
- Toggle auto-mode (trainer promotes/demotes based on CPU)
- Pause/resume training
- Hot-swap knobs without restart (e.g. `BATCH_SIZE`, `MCTS_VISITS_SELFPLAY`)
- Watch model-vs-model games and puzzle drills in the spectate modal

Full API reference is in `train/TUNING_REFERENCE.md` ("Control Server +
Dashboards" section).

## Hot-swap knobs (Fase 0)

Most training constants are changeable at runtime:

```python
trainer.set_knob("BATCH_SIZE", 512)
trainer.set_knobs({"MCTS_VISITS_SELFPLAY": 400, "NUM_SELF_PLAY_WORKERS": 12})
print(trainer.list_hot_knobs())   # what's tunable
```

Or via HTTP:

```bash
curl -X POST http://127.0.0.1:7860/api/knobs -d '{"knobs":{"BATCH_SIZE":512}}'
```

Immediate knobs (`BATCH_SIZE`, loss weights, puzzle batches) apply on the
next training step. Deferred knobs (`MCTS_VISITS_SELFPLAY`,
`NUM_SELF_PLAY_WORKERS`, `REPLAY_BUFFER_MAX_SIZE`, eval cadence) apply
at the next round boundary.

## Spectate + puzzle sidecar (Fase 4/4b)

Watch model-vs-model games or run puzzle drills from the dashboard or
the HTTP API:

```bash
# Model vs model
curl -X POST http://127.0.0.1:7860/api/matches -d '{
  "type": "model", "params": {"white": "baseline", "black": "attack", "visits": 200}
}'

# Puzzle drill (random from sidecar)
curl -X POST http://127.0.0.1:7860/api/matches -d '{
  "type": "puzzle", "params": {"visits": 100}
}'
```

Events stream over `/api/matches/stream` (SSE). Puzzle drills need a
sidecar — the cached tensors don't preserve FENs:

```bash
cd train
python -m league.puzzle_sidecar
# or from the repo root:
python train/build_puzzle_sidecar.py
```

This writes `train/cache/puzzles_meta.pkl` (~300MB for the full Lichess
DB). The sidecar is loaded lazily on the first drill request.

---

**The Rule**: Only MCTS self-play improves models. Everything else is measurement, cold-start reduction, or regression detection.
