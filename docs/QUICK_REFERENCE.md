# League Training: Quick Reference Card

## Start Training

```bash
cd chess
python league/main.py
```

## Monitor (Real-Time)

```bash
tail -f logs/metrics.log
```

## Check Status

```bash
# Model checkpoints
ls -lh checkpoints/

# Metrics snapshot
cat logs/metrics_round_0.json | grep -A 20 '"baseline"'

# Training log
tail -20 logs/league_training.log
```

## Stop Training

Press `Ctrl+C` in the terminal running `league/main.py`

## Key Metrics to Watch

| Metric | Good | Bad | What It Means |
|--------|------|-----|---------------|
| `loss` | Decreasing over time | Flat or increasing | Model learning |
| `recent_loss` | < 0.5 after warm-up | > 1.0 | Training quality |
| `buffer_fill_ratio` | > 0.8 | < 0.3 | Data generation rate |
| `avg_game_length` | 40-100 moves | Always < 30 or > 150 | Game quality |

## Configuration (Edit league/league_trainer.py)

```python
class LeagueTrainer:
    # Parallelism
    NUM_SELF_PLAY_WORKERS = 6              # Increase if CPU idle
    GAMES_PER_WORKER_PER_ROUND = 10        # Increase for more data
    
    # Training
    BATCH_SIZE = 256                       # Decrease if GPU OOM
    TRAINING_STEPS_PER_ROUND = 50          # Increase if GPU idle
    
    # MCTS
    MCTS_VISITS_TRAINING = 800             # Increase for better data
    TEMPERATURE = 1.0                      # Lower = more greedy
    
    # Frequency
    CHECKPOINT_EVERY_N_ROUNDS = 5          # Less frequent = faster
    EVAL_EVERY_N_ROUNDS = 10               # Less frequent = faster
```

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| GPU out of memory | Reduce `BATCH_SIZE` from 256 to 128 |
| Workers crash | Check if MCTS works: `python -c "from train.mcts import MCTS; print('OK')"` |
| Training slow | Increase `NUM_SELF_PLAY_WORKERS` from 6 to 8 |
| Loss not decreasing | Check learning rate or increase `BATCH_SIZE` |
| No games being generated | Reduce `NUM_SELF_PLAY_WORKERS` (workers timing out) |

## File Structure

```
league/                         # Main system (run this)
├── main.py                     # Entry point
├── league_trainer.py           # Orchestrator
├── replay_buffer.py            # Data storage
├── self_play_worker.py         # MCTS worker
├── evaluator.py                # Regression detection
├── monitoring.py               # Metrics
└── __init__.py

docs/                           # Read these
├── README.md                   # Start here
├── QUICKSTART.md               # Copy-paste examples
├── ARCHITECTURE.md             # How it works
├── MODULE_REFERENCE.md         # API docs
├── THE_RULE.md                 # Why it works
└── INTEGRATION.md              # Migration guide

checkpoints/                    # Auto-saved models
├── baseline_step_0.pt
├── baseline_step_5.pt
└── ...

logs/                           # Auto-saved metrics
├── metrics.log                 # Real-time log
├── metrics_round_0.json        # Snapshots
└── league_training.log         # Info log
```

## Performance Targets (RTX 5080)

| Phase | Time | GPU | CPU |
|-------|------|-----|-----|
| Self-play (60 games) | 2-3 min | 0% | 100% |
| Training (50 steps) | 30-60 sec | 100% | 10% |
| **Total round** | **3.5-4.5 min** | Balanced | Balanced |

## The Rule (Critical!)

> **Only MCTS self-play improves models long-term.**

✅ Allowed:
- Self-play with MCTS
- Training on MCTS data
- Evaluation (for measuring, not training)
- Metrics collection

❌ Not allowed in main loop:
- Policy distillation from other models
- Supervised learning on puzzles
- Ensemble voting
- Mixed opponent buffers

## Common Commands

```bash
# Start training
python league/main.py

# View live metrics
tail -f logs/metrics.log

# Check recent models
ls -lh checkpoints/ | tail -10

# View performance metrics
jq '.variants | keys[]' logs/metrics_round_*.json | head -5

# Count saved checkpoints
ls checkpoints/*.pt | wc -l

# See how many games played
grep "games" logs/metrics.log | tail -1

# Stop cleanly (no kill -9!)
# Press Ctrl+C in the league/main.py terminal
```

## Debug Mode

Enable verbose logging:

```python
# In league/main.py, add after imports:
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

## Evaluate a Checkpoint Against Stockfish

```bash
python tools/evaluate_with_stockfish.py checkpoints/baseline_step_50.pt
```

## Next Steps

1. ✅ Run `python league/main.py`
2. ✅ Monitor `tail -f logs/metrics.log`
3. ✅ Wait 10 rounds, check loss trend
4. ✅ If good: let it run longer
5. ✅ If bad: check troubleshooting table
6. ✅ Read [ARCHITECTURE.md](docs/ARCHITECTURE.md) for details

---

**The system is ready. Start it and let MCTS self-play do the work.**
