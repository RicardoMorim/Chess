# Module Reference

## league_trainer.py

Main orchestrator for league training.

### LeagueTrainer

```python
class LeagueTrainer:
    """Main training orchestrator for parallel league training."""
```

#### Key Methods

**`__init__(checkpoint_dir, log_dir, device)`**
- Initializes trainer with directories and device
- Creates metrics collector and evaluator

**`initialize_models(model_constructor, model_configs)`**
- Creates N model variants with different configs
- Sets up optimizers (SGD) and replay buffers for each
- Must be called before `run()`

**`generate_self_play(variant)`**
- Launches N parallel workers on CPU
- Each plays GAMES_PER_WORKER_PER_ROUND games
- Collects games to replay buffer
- Returns: number of games generated

**`train_model(variant)`**
- Trains model on replay buffer data
- Runs TRAINING_STEPS_PER_ROUND SGD steps
- Samples BATCH_SIZE positions randomly from buffer
- Returns: average loss

**`save_checkpoint(variant, step)`**
- Saves model state dict to file
- Registers with evaluator for future tests
- Returns: path to checkpoint

**`run(max_rounds=None)`**
- Main training loop
- Per round: self-play → train → checkpoint → evaluate
- Logs metrics, saves snapshots
- Runs indefinitely if max_rounds=None

#### Configuration

```python
VARIANTS = ["baseline", "attack", "est"]
NUM_SELF_PLAY_WORKERS = 6
GAMES_PER_WORKER_PER_ROUND = 10
BATCH_SIZE = 256
TRAINING_STEPS_PER_ROUND = 50
CHECKPOINT_EVERY_N_ROUNDS = 5
EVAL_EVERY_N_ROUNDS = 10

MCTS_VISITS_TRAINING = 800
TEMPERATURE = 1.0
C_PUCT = 4.0
DIRICHLET_ALPHA = 0.3
```

---

## replay_buffer.py

Thread-safe FIFO replay buffer for each model.

### ReplayBuffer

```python
class ReplayBuffer:
    """FIFO replay buffer with thread-safe operations."""
```

#### Key Methods

**`add_game(game_trajectory)`**
- Adds complete game to buffer
- `game_trajectory`: list of (position, policy, value) tuples
- Thread-safe, automatic FIFO eviction when full

**`sample(batch_size)`**
- Returns random batch of (positions, policies, values)
- Raises ValueError if batch_size > buffer size

**`__len__()`**
- Returns current number of positions

**`is_ready(min_size=256)`**
- Returns True if buffer has enough data for training

**`get_stats()`**
- Returns dict: size, capacity, fill_ratio, value_mean, value_std

#### Example

```python
buffer = ReplayBuffer(max_size=200_000)

# Add games from self-play
buffer.add_game(game_trajectory)

# Sample during training
if buffer.is_ready():
    positions, policies, values = buffer.sample(256)
    # Use for training...

# Monitor
stats = buffer.get_stats()
print(f"Buffer: {stats['size']}/{stats['capacity']} "
      f"({stats['fill_ratio']:.1%})")
```

---

## self_play_worker.py

CPU worker that generates games via MCTS.

### self_play_worker()

```python
def self_play_worker(
    model_state_dict,
    model_constructor,
    num_games,
    device,
    result_queue,
    model_config=None,
    mcts_config=None,
    worker_id=0,
):
    """Generate self-play games and queue to parent process."""
```

#### Parameters

- `model_state_dict`: Frozen model weights (dict)
- `model_constructor`: Callable that creates model instance
- `num_games`: Number of games to play
- `device`: "cpu" or "cuda"
- `result_queue`: multiprocessing.Queue for results
- `model_config`: Dict with model hyperparams (optional)
- `mcts_config`: Dict with MCTS hyperparams (optional)
- `worker_id`: For logging

#### Returns

- None (results queued as dicts: `{"game_data": [...], "worker_id": ...}`)

#### Default Configs

```python
model_config = {
    "input_channels": 22,
    "num_blocks": 15,
    "channels": 256,
}

mcts_config = {
    "num_visits": 800,
    "temperature": 1.0,
    "c_puct": 4.0,
    "dirichlet_alpha": 0.3,
    "add_noise": True,
}
```

#### Example

```python
from multiprocessing import Queue, Process
from train.models import create_model

queue = Queue()
p = Process(
    target=self_play_worker,
    args=(
        model.state_dict(),
        create_model,
        10,
        "cpu",
        queue,
    )
)
p.start()

for _ in range(10):
    result = queue.get()
    game_data = result["game_data"]
    # Use game_data...

p.join()
```

---

## monitoring.py

Centralized metrics collection and logging.

### MetricsCollector

```python
class MetricsCollector:
    """Collects and aggregates metrics from all league components."""
```

#### Key Methods

**`record_metric(name, value, variant=None)`**
- Record a time-series metric
- Thread-safe

**`record_counter(name, increment=1, variant=None)`**
- Increment a counter

**`set_gauge(name, value, variant=None)`**
- Set current gauge value

**`record_self_play_game(variant, game_length, outcome)`**
- Record self-play statistics

**`record_training_step(variant, loss, policy_loss, value_loss, learning_rate)`**
- Record training metrics

**`record_evaluation(variant, opponent, result, elo_change=0)`**
- Record evaluation results

**`record_buffer_stats(variant, buffer_size, capacity, value_mean, value_std)`**
- Record buffer health

**`get_summary()`**
- Get current metrics snapshot (dict)

**`save_checkpoint(name=None)`**
- Save metrics to JSON file

**`log_summary(prefix="")`**
- Log all metrics to logger

#### Example

```python
metrics = MetricsCollector("logs")

# During training
for step in range(50):
    loss = train_step(...)
    metrics.record_training_step(
        "baseline",
        loss,
        policy_loss,
        value_loss,
        lr
    )

# Periodically
metrics.log_summary(f"Round {round}")
metrics.save_checkpoint(f"round_{round}")

# Check buffer
buffer_stats = buffer.get_stats()
metrics.record_buffer_stats("baseline", **buffer_stats)
```

---

## evaluator.py

Low-frequency evaluation against frozen checkpoints.

### Evaluator

```python
class Evaluator:
    """Evaluates models against frozen opponent checkpoints."""
```

#### Key Methods

**`register_checkpoint(variant, step, model_state)`**
- Register a checkpoint for future evaluation

**`evaluate_matchup(current_model, current_variant, opponent_checkpoint, opponent_model)`**
- Run evaluation games
- Uses frozen MCTS (temperature=0.1, no noise)
- Returns: dict with wins, draws, ELO diff

**`get_regression_report(current_variant, threshold_elo_loss=50)`**
- Check if recent performance degraded
- Returns: dict with regression flag and worst matchup

#### Example

```python
evaluator = Evaluator(device="cuda", eval_games_per_matchup=20)

# Register checkpoints
evaluator.register_checkpoint("baseline", 0, model.state_dict())

# Later, evaluate current vs old
result = evaluator.evaluate_matchup(
    current_model, "baseline",
    "baseline_step_0",
    old_model
)

print(f"Win rate: {result['current_win_rate']:.1%}")
print(f"ELO diff: {result['estimated_elo_diff']:.1f}")
```

---

## Integration Points

### With train/models.py

```python
from train.models import create_model

trainer.initialize_models(
    model_constructor=create_model,
    model_configs={...}
)
```

### With train/mcts.py

```python
from train.mcts import MCTS

mcts = MCTS(model, device, num_visits=800, temperature=1.0, ...)
policy, move = mcts.search(board)
```

### With train/data.py

```python
from train.data import board_to_tensor, get_move_index

position = board_to_tensor(board, input_channels=22)
move_idx = get_move_index(move)
```

---

## Performance Characteristics

### Self-Play Worker

- **CPU**: Single core, 100% for MCTS
- **GPU**: Minimal (one batch inference per position)
- **Memory**: ~500MB (model + MCTS tree)
- **Speed**: ~40-80 games/hour depending on hardware

### Training

- **GPU**: 100% utilization with batch_size=256
- **Memory**: ~8GB for model + batch
- **Speed**: ~50 steps/second (varies by model size)

### Buffer

- **Memory**: 200k positions × channels × 8 × 8 × 4 bytes ≈ 1-2GB
- **Speed**: O(1) add/sample with thread lock

### Metrics

- **Memory**: O(num_metrics), negligible
- **Speed**: O(1) per record_*() call
- **I/O**: ~1MB per checkpoint

---

## Common Patterns

### Add Custom Metric

```python
metrics.record_metric("custom/metric_name", value, variant="baseline")
```

### Monitor Buffer Health

```python
for variant in trainer.VARIANTS:
    stats = trainer.buffers[variant].get_stats()
    if stats["fill_ratio"] < 0.5:
        print(f"WARNING: {variant} buffer not filling fast enough")
```

### Custom Training Loop

```python
for round in range(max_rounds):
    # Self-play
    for variant in trainer.VARIANTS:
        trainer.generate_self_play(variant)
    
    # Custom training logic
    for variant in trainer.VARIANTS:
        for step in range(100):  # More steps
            trainer.train_model(variant)
    
    # Checkpoint less frequently
    if round % 10 == 0:
        for variant in trainer.VARIANTS:
            trainer.save_checkpoint(variant, round)
```

---

**For more details, see ARCHITECTURE.md and QUICKSTART.md**
