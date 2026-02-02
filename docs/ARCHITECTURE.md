# League Training Architecture

## Overview

This is a **production-ready parallel league training system** for chess AI that follows the fundamental principle:

> **THE RULE: Only self-play with MCTS improves models long-term. Everything else exists only to reduce cold start, catch regressions, or measure strength.**

The architecture separates concerns into four independent layers:

1. **Self-Play Workers** (CPU-bound, parallel) - Generate training data via MCTS
2. **Training Loop** (GPU-bound, sequential) - Optimize on batches from replay buffer
3. **Replay Buffers** (persistent per-model) - Decouple generation from training
4. **Evaluation & Monitoring** (low-frequency) - Track progress without interfering

## Key Improvements Over Naive Approaches

| Problem | Previous Approach | New Solution | Why It Works |
|---------|-------------------|--------------|--------------|
| CPU idle | Sequential self-play | N parallel workers | Fully utilizes CPU cores |
| GPU idle | No batching | Large batch training | GPU stays busy |
| Forgetting | Mixed buffers | Per-model FIFO buffers | Stable learning dynamics |
| Co-adaptation | All variants play each other | Checkpointed opponents | League structure prevents collapse |
| Unmeasured progress | No evaluation | Low-freq checkpointed tests | Catch regressions early |
| Blind training | No metrics | Centralized monitoring | Real-time diagnostics |

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│  LEAGUE TRAINER (Main Process)                          │
│  - Orchestrates rounds                                  │
│  - Checkpoints models                                   │
│  - Collects metrics                                     │
└─────────────────┬───────────────────────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
      ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐  VARIANT 1: "baseline"
│ SP Work  │ │ SP Work  │ │ SP Work  │  - 18 input channels
│ (CPU)    │ │ (CPU)    │ │ (CPU)    │  - 15 residual blocks
│ MCTS800  │ │ MCTS800  │ │ MCTS800  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     └────────────┼────────────┘
                  │ (game queues)
                  ▼
         ┌─────────────────┐
         │ Replay Buffer   │
         │ (200k positions)│
         └────────┬────────┘
                  │ (sample batches)
                  ▼
         ┌─────────────────┐
         │ GPU Training    │
         │ (SGD, 256 batch)│
         │ 50 steps/round  │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Checkpoint save │
         │ (every 5 rounds)│
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ Metrics collect │
         │ (every round)   │
         └─────────────────┘

(Same pipeline for variants: "attack", "est")
```

## File Structure

```
league/
├── replay_buffer.py      # Per-model FIFO buffers with thread-safety
├── self_play_worker.py   # CPU worker: generates games via MCTS
├── league_trainer.py     # Main orchestrator (this is the control hub)
├── evaluator.py          # Low-freq evaluation vs checkpoints
└── monitoring.py         # Metrics collection + logging

checkpoints/
├── baseline_step_0.pt    # Saved model states
├── baseline_step_5.pt
├── attack_step_0.pt
└── ...

logs/
├── metrics.log           # All metric events
├── metrics_round_0.json  # Periodic snapshots
└── metrics_round_5.json
```

## The Rule in Action

### ✅ What DOES Improve Models Long-Term

- **Self-play with MCTS**: Only source of improvement. Workers run frozen MCTS on latest model weights.
- **Replay buffer sampling**: Trains on this MCTS-generated data, nothing else.

### ❌ What MUST NOT Improve Models

- ~~Policy distillation~~: Disabled. Models only learn from MCTS values.
- ~~Mixed opponent buffers~~: Disabled. Each variant has isolated buffer.
- ~~Supervised bootstrapping~~: Separate phase only (reduces cold start), removed from main loop.
- ~~Teacher-student pipelines~~: Disabled. No inter-variant improvement.

### 🔍 Measurement-Only Activities (Don't Block Training)

- **Evaluation**: Run vs frozen checkpoints at low frequency (every 10 rounds)
- **Regression detection**: Compare recent vs baseline checkpoints
- **Metrics collection**: Track losses, game lengths, buffer fill ratio

## How to Use

### 1. Initialize

```python
from league.league_trainer import LeagueTrainer
from train.models import create_model  # Your model factory

trainer = LeagueTrainer(
    checkpoint_dir="checkpoints",
    log_dir="logs",
    device="cuda"
)

trainer.initialize_models(
    model_constructor=create_model,
    model_configs={
        "baseline": {"input_channels": 18},
        "attack": {"input_channels": 22},
        "est": {"input_channels": 22},
    }
)
```

### 2. Run Training Loop

```python
trainer.run(max_rounds=1000)
```

Each round:
1. **Self-play**: 6 workers × 10 games = 60 games/variant in parallel (~2-3 min)
2. **Training**: 50 steps of SGD on 256-batch samples (~30-60 sec)
3. **Checkpointing**: Every 5 rounds (~10 sec)
4. **Evaluation**: Every 10 rounds (~5-10 min, runs in background)
5. **Metrics**: Logged to `logs/metrics.log`, snapshots saved every round

### 3. Monitor Training

```python
# View real-time metrics
tail -f logs/metrics.log

# View periodic snapshots
cat logs/metrics_round_50.json

# Check checkpoint progress
ls -lah checkpoints/ | tail -20
```

## Performance Expectations

### Hardware: RTX 5080

| Component | Time | CPU | GPU | Bottleneck |
|-----------|------|-----|-----|-----------|
| Self-play (60 games) | ~2-3 min | 100% (6 cores) | 0% | CPU parallelism |
| Training (50 steps) | ~30-60 sec | 0% | 100% | Batch processing |
| Checkpoint save | ~5 sec | 20% | 10% | I/O |
| **Total per round** | **~3.5-4.5 min** | Balanced | Balanced | Neither |

**Expected throughput**: ~15 games/minute × 800 MCTS visits = **~12,000 MCTS nodes/second per core**.

### Scaling

- Add more workers: Self-play time stays constant, CPU usage increases linearly
- Larger batches: Training time increases, GPU utilization stays ~100%
- More training steps: Training time increases, each step processes independent batch

## Monitoring & Metrics

### Automatic Collection

Every round, the system tracks:

```json
{
  "round": 42,
  "timestamp": "2026-02-02T15:30:00",
  "variants": {
    "baseline": {
      "games": 600,
      "train_steps": 2100,
      "buffer_size": 184723,
      "buffer_fill_ratio": 0.923,
      "recent_loss": 0.324,
      "avg_game_length": 78.5
    },
    "attack": {
      "games": 600,
      "train_steps": 2100,
      "buffer_size": 199456,
      "buffer_fill_ratio": 0.997,
      "recent_loss": 0.401,
      "avg_game_length": 82.1
    }
  }
}
```

### Regression Detection

If any variant shows:
- **Loss not decreasing** over 20 rounds → Check learning rate
- **Buffer fill ratio < 0.5** → Self-play too slow (more workers needed)
- **Evaluation ELO drop > 50** → Regression detected (revert to previous checkpoint)

## Custom Model Integration

To use your own model:

```python
# 1. Ensure your model has:
#    - __init__(self, input_channels, num_blocks, channels)
#    - forward(self, x) -> (policy_logits, value)

# 2. Create a factory function:
def create_model(input_channels=22, **kwargs):
    return ChessNet(input_channels=input_channels, **kwargs)

# 3. Pass to trainer:
trainer.initialize_models(
    model_constructor=create_model,
    model_configs={
        "baseline": {"input_channels": 18},
        "attack": {"input_channels": 22},
        "est": {"input_channels": 22},
    }
)
```

## Common Issues & Solutions

### "Workers stuck" / "No games generated"

- Check if MCTS.search() works on model
- Verify board_to_tensor() output shape matches model input
- Check for GPU memory exhaustion (workers run on CPU, but model inference needs VRAM)

### "Training loss not decreasing"

- Check learning rate (default 0.01 may be too high)
- Verify buffer has good value distribution (not all 0s or 1s)
- Check if MCTS policy is reasonable (not uniform over all squares)

### "Memory grows unbounded"

- Replay buffers have max_size=200k, automatically FIFO
- If still growing, check for queue leaks in worker processes
- Monitor `/proc/[pid]/mem` for each worker

### "GPU underutilized"

- Increase BATCH_SIZE (currently 256)
- Increase TRAINING_STEPS_PER_ROUND (currently 50)
- Decrease self-play frequency (fewer rounds generate self-play data)

## Next Steps (Future Enhancements)

Not needed for core training, but useful later:

- [ ] **PSRO (Policy-Space Response Oracle)**: Automatically discover new strategies
- [ ] **Elo tracking**: Maintain historical strength ratings
- [ ] **Web dashboard**: Real-time metrics visualization
- [ ] **Distributed training**: Multi-GPU support across machines
- [ ] **Warm-starting**: Load pretrained models instead of random init
- [ ] **Hyperparameter search**: Auto-tune learning rate, batch size, visits

## References

This architecture is inspired by:

- **AlphaZero** (Silver et al., 2017): Self-play + MCTS + neural networks
- **AlphaGo**: Replay buffer for training stability
- **League training** concepts from OpenAI Five: Multiple parallel agents
- **Best practices** from modern game AI (Leela Chess Engine, etc.)

---

**Rule reminder**: If a component doesn't serve self-play improvement, regression detection, or strength measurement, it doesn't belong in this loop.
