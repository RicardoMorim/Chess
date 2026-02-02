# The Rule: Self-Play MCTS as Sole Improvement Mechanism

## Core Principle

> **Only self-play with MCTS is allowed to improve models long-term. Everything else exists only to:**
> 1. **Reduce cold start** (bootstrap phase)
> 2. **Catch regressions** (evaluation)
> 3. **Measure strength** (monitoring)

## Why This Rule Exists

### The Problem

In early attempts at chess AI training, it's tempting to use many signals:
- Teacher-student distillation
- Supervised learning on puzzles
- Policy imitation from engines
- Ensemble predictions
- Multi-model co-training

This leads to:

| What Happens | Why It's Bad |
|--------------|-------------|
| **Model collapse** | Multiple signals contradict each other |
| **Convergence to local optima** | Model learns to imitate rather than improve |
| **Catastrophic forgetting** | New data overwrites learned patterns |
| **Uncontrolled co-adaptation** | Models overfit to each other's weaknesses |
| **Unmeasurable progress** | Can't tell what actually helped |

### AlphaZero's Insight

AlphaZero proved that **a single, consistent training signal is far more powerful**:

1. Generate games with MCTS (exploration + evaluation)
2. Train neural network on this data (supervised learning)
3. Improve move selection (better policy + value)
4. Return to step 1

This creates a **positive feedback loop** where each component strengthens the others.

## What This Looks Like in Practice

### ✅ Allowed: Self-Play MCTS

```python
# Generate training data
for game in range(num_games):
    board = chess.Board()
    while not game_over:
        # MCTS: tree search using CURRENT model weights
        policy = mcts.search(board, num_visits=800)
        move = select_move(policy)
        board.push(move)
    
    # Record (state, policy, outcome)
    game_data.append((state, policy_from_mcts, outcome))

# Train on MCTS data
for batch in replay_buffer.sample():
    policy_pred, value_pred = model(batch_states)
    loss = cross_entropy(policy_pred, mcts_policy) + mse(value_pred, outcome)
    optimizer.step()
```

This closes the loop: **Model improves → MCTS searches better → Data improves → Model improves more**

### ❌ Not Allowed: Policy Distillation

```python
# DON'T DO THIS
baseline_model = load_checkpoint("baseline_step_0")
current_model = load_checkpoint("current_step_42")

# Mix predictions from both models
mixed_policy = 0.5 * baseline_model.policy + 0.5 * current_model.policy

# Train on mixed policy
loss = cross_entropy(current_model.policy, mixed_policy)
```

Why not:
- No clear improvement signal
- Baseline might be wrong, but weighted equally
- Breaks the self-play → improvement → MCTS feedback loop
- Creates dependency on checkpoint selection

### ❌ Not Allowed: Supervised Learning on Puzzles

```python
# DON'T DO THIS (in main training loop)
for puzzle_position in puzzle_dataset:
    engine_move = stockfish.best_move(puzzle_position)
    model_move = model.select_move(puzzle_position)
    
    loss = cross_entropy(model.policy, one_hot(engine_move))
    optimizer.step()
```

Why not:
- Teaches model to imitate engines, not to play better
- Model learns "memorized" patterns, not generalizable skill
- Once puzzle is memorized, no more improvement signal
- Takes training time away from self-play

## The Architecture Enforces the Rule

### Where Improvement Happens

```
Self-Play Workers (CPU)
    ↓ generates games via MCTS
Replay Buffer
    ↓ stores position, policy, value triples
Training Loop (GPU)
    ↓ optimizes model on this data
Model Checkpoint
    ↓ updated weights loaded into MCTS
Self-Play Workers (back to step 1)
```

### Where It Doesn't Happen

```
Evaluation
    ↓ Tests vs old checkpoints
    ↓ Doesn't update training models
    ✗ No improvement

Monitoring
    ↓ Tracks metrics
    ✗ Doesn't update weights

Bootstrap Phase
    ↓ Optional: supervised learning to reduce cold start
    ✓ Allowed because it's ONE-TIME, not in main loop
```

## Enforcement Mechanisms

### 1. Isolated Replay Buffers

```python
# Each variant has its own buffer
buffers = {
    "baseline": ReplayBuffer(),
    "attack": ReplayBuffer(),
    "est": ReplayBuffer(),
}

# Data flows: Self-Play → Buffer → Training
# NOT: Self-Play → Shared Buffer ← Other Models
```

**Why**: Prevents models from mixing data or imitating each other.

### 2. No Optimizer Access in Workers

```python
def self_play_worker(...):
    # Workers are stateless
    model = load_model_state()
    for game in games:
        data = play_game_with_mcts(model)
        result_queue.put(data)
    
    # No optimizer.step() here
    # No access to replay buffer
    # No weight updates
```

**Why**: Keeps generation (CPU) and training (GPU) separated.

### 3. Checkpointed Opponents

```python
# Evaluation uses OLD checkpoints
old_baseline = load_checkpoint("baseline_step_0")
current_baseline = load_checkpoint("baseline_step_42")

# Play against old version
result = evaluate(current_baseline, old_baseline)

# If worse: regression detected
# If better: confirmed improvement
```

**Why**: Prevents co-adaptation where all models drift in same direction.

### 4. Centralized Metrics (Read-Only)

```python
# Monitoring doesn't change training
metrics.record_metric("loss", loss_value)
metrics.record_metric("win_rate", eval_result)

# These values are logged, not fed back into training
```

**Why**: Prevents feedback loops where metrics affect training decisions.

## Exception: Bootstrap Phase

The only allowed exception to the rule is **one-time supervised learning to reduce cold start**:

```python
# OK: Before league training starts
for epoch in range(epochs):
    for puzzle_batch in puzzle_loader:
        engine_move = get_engine_move(puzzle_batch)
        
        policy_pred = model(puzzle_batch)
        loss = cross_entropy(policy_pred, engine_move)
        
        optimizer.step()

# Save checkpoint
save_checkpoint("warm_start.pt")

# Now start league training
# NO MORE supervised learning in the main loop
```

This is allowed because:
- ✓ Reduces cold start (self-play is very slow initially)
- ✓ Happens once, before main training
- ✓ After this, only MCTS improves the model

## How to Check if Code Violates the Rule

### Red Flags

❌ **Model being trained on data not from current self-play**
```python
loss = cross_entropy(model.policy, old_checkpoint.policy)  # NO
```

❌ **Mixing predictions from multiple models**
```python
mixed_policy = 0.5 * model1 + 0.5 * model2
loss = cross_entropy(current_model, mixed_policy)  # NO
```

❌ **Using external data (puzzles, engines) in main loop**
```python
# In main training loop:
for puzzle in puzzles:
    loss = supervised_loss(model, puzzle)  # NO (unless bootstrap phase)
```

❌ **Models learning from each other's outputs**
```python
# In main training loop:
teacher_policy = teacher_model(state)
student_loss = cross_entropy(student_model(state), teacher_policy)  # NO
```

### Green Lights

✅ **All training data from current self-play**
```python
policy_from_mcts = mcts.search(board)
loss = cross_entropy(model_policy, policy_from_mcts)  # YES
```

✅ **Evaluation doesn't update weights**
```python
result = evaluate(current_model, old_checkpoint)
# result is logged, not used for training
```

✅ **Replay buffer stores MCTS data only**
```python
for position, policy_from_mcts, outcome in game_trajectory:
    buffer.add((position, policy_from_mcts, outcome))
```

✅ **Each model trains on its own buffer**
```python
for variant in ["baseline", "attack", "est"]:
    buffer = buffers[variant]  # Own buffer
    batch = buffer.sample()
    train(models[variant], batch)
```

## FAQ

### Q: Why not use ensemble predictions during training?

**A**: Ensembles mask which component actually improved. If performance goes up, you don't know if it was the model update or just changing ensemble weights. Self-play provides clear signal: better model → better MCTS → better data → model improves.

### Q: Why not distill from a stronger external engine?

**A**: Because then the model learns to imitate the engine, not to play better. You want the model to **improve** by playing against itself, not to **match** an external player. After enough self-play, your model will be stronger than the engine anyway.

### Q: Why not use multiple loss functions (policy + value + regularization)?

**A**: Multiple losses are fine if they're all computed from MCTS-generated data. What matters is the **source of improvement signal**, not the number of terms. As long as the data comes from self-play, it's allowed.

### Q: Why can we use puzzles for cold start but not later?

**A**: Because cold start is special—self-play is very slow initially with a random model. Supervised learning on puzzles gets you moving faster. But once self-play starts, it becomes the dominant improvement signal and should stay that way.

### Q: Can we use supervised learning to fix specific weaknesses?

**A**: No. If the model has a weakness, let self-play fix it. MCTS will naturally expose and punish weaknesses, driving the model to improve. Supervised learning would teach it to memorize the weakness fix, not to generalize better.

---

## References

- **AlphaZero** (Silver et al., 2017): Original paper introducing this approach
- **AlphaGo** (Schultze et al., 2016): First application of self-play + MCTS + neural networks
- **Game Theory**: Self-play is essentially the game reaching equilibrium through repeated interaction
- **Information Theory**: Single, consistent signal is more informative than noisy ensemble

---

**Bottom Line**: Trust the self-play loop. It's proven to work. Everything else should be subordinate to keeping that loop clean and efficient.
