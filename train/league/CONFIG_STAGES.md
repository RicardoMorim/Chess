# League Training Configuration Stages

This document explains the progression from bootstrap validation to production-scale league training.

## Critical Design Principle: Two MCTS Budgets

**Self-play must be FAST. Evaluation must be DEEP.**

```
Self-play (VOLUME):   16 visits  → Generate 20-30 games per round
Evaluation (QUALITY): 64 visits  → Compare models meaningfully

NOT the other way around.
```

**Why?** Training improves the model, not more MCTS in self-play. You want:
- Lots of data (16-24 visits per move is fine)
- Fast generation (no wall-clock timeouts)
- GPU training to do the heavy lifting

---

## Stage 1: Bootstrap Validation ✅ (COMPLETED)

**Purpose**: Verify end-to-end pipeline correctness without wasting compute

**Configuration**:
```python
NUM_SELF_PLAY_WORKERS = 1
GAMES_PER_WORKER_PER_ROUND = 1
BATCH_SIZE = 64
TRAINING_STEPS_PER_ROUND = 5
MCTS_VISITS_SELFPLAY = 16  # Fast
MCTS_VISITS_EVAL = 32
```

**Expected Runtime**: 
- ~3-4 minutes per round (1 game × 3 models)
- Games complete normally (resignations + reasonable move counts)

**Validation Criteria**:
- ✅ Workers start and complete without crashes
- ✅ Game data flows to replay buffers
- ✅ Training runs without NaN/Inf losses
- ✅ Metrics track correctly (no buffer mismatch)
- ✅ Checkpoints save successfully

**Issues Fixed**:
1. ACTION_SPACE_SIZE import chain
2. Metrics double-prefix bug (counters/gauges)
3. result_queue parameter passing
4. Game time/move limits
5. Early resignation logic

---

## Stage 2: Intermediate Parallelism ⏳ (CURRENT)

**Purpose**: Enable real parallelism with correct MCTS budgeting

**Configuration**:
```python
NUM_SELF_PLAY_WORKERS = 4
GAMES_PER_WORKER_PER_ROUND = 5
BATCH_SIZE = 128
TRAINING_STEPS_PER_ROUND = 10
MCTS_VISITS_SELFPLAY = 16  # FAST generation (no timeouts)
MCTS_VISITS_EVAL = 64      # Quality comparisons
```

**Expected Runtime**:
- ~8-12 minutes per round (20 games × 3 models = 60 games total)
- Games complete without timeouts (16 visits = ~3-5 sec per move)
- Average game length 40-80 moves (resignations active)

**Key Insight (CRITICAL)**:
- ❌ DO NOT use 64 visits for self-play (causes 45-55 move timeouts)
- ✅ DO use 16 visits for self-play (fast, data-generation focused)
- ✅ DO use 64 visits only for evaluation (model comparison)

**Per-Move Cost Analysis**:
```
16 visits × policy+value inference = ~300ms per move (CPU, Python)
50 moves × 16 = ~2500ms = 2.5 sec per game (reasonable)

64 visits × policy+value inference = ~1200ms per move
50 moves × 64 = ~60,000ms = 60 sec per move... TIMEOUT!
```

**Validation Criteria**:
- ✅ Buffer grows steadily (600+ positions per round, no timeouts)
- ✅ Loss decreases over 5-10 rounds
- ✅ Games complete in <2 min per game
- ✅ No worker crashes or queue deadlocks
- ✅ GPU training improves value estimates

**Parallelism Impact**:
- Self-play: 1 game → 20 games (20x throughput)
- GPU remains idle during self-play (CPU-bound)
- Training saturates GPU (Phase 2)

---

## Stage 3: Production Scale (FUTURE)

**Purpose**: Full AlphaZero-quality MCTS and compute saturation

**Configuration**:
```python
NUM_SELF_PLAY_WORKERS = 6-8  # Depends on CPU cores
GAMES_PER_WORKER_PER_ROUND = 10
BATCH_SIZE = 256
TRAINING_STEPS_PER_ROUND = 20
MCTS_VISITS_SELFPLAY = 800   # Still fast enough if GPU-batched
MCTS_VISITS_EVAL = 1600
```

**Prerequisites**:
- Stage 2 running stably for 50+ rounds
- Loss consistently decreasing
- Replay buffer at >10k positions
- No memory leaks or crashes
- **GPU-accelerated MCTS** (not CPU-bound Python)

**Optimizations Needed**:
- MCTS inference batching across workers
- Async self-play (don't wait for all workers)
- Mixed precision training (fp16)
- Gradient accumulation for larger effective batch sizes

---

## Monitoring Checklist

### Per Round
- [ ] All workers complete successfully
- [ ] Buffer size increases
- [ ] Loss finite (no NaN/Inf)
- [ ] No game timeouts
- [ ] Checkpoints save

### Per 10 Rounds
- [ ] Loss trend downward (5-10% decrease every 5 rounds initially)
- [ ] Game length decreases (better play, more resignations)
- [ ] No MCTS failures
- [ ] Metrics JSON readable

### Per 100 Rounds
- [ ] Run evaluation vs checkpoints
- [ ] Check for overfitting (value_std should not collapse)
- [ ] Validate model outputs (not degenerate)

---

## Troubleshooting

**Issue**: "Game timeout after 45-55 moves" (MOST COMMON)
- **Root Cause**: MCTS_VISITS too high for self-play
- **Fix**: Use MCTS_VISITS_SELFPLAY=16, MCTS_VISITS_EVAL=64 (not both high)
- **Verify**: 16 visits should allow games to complete in <2 min

**Issue**: Workers hang indefinitely
- **Cause**: Deadlock in queue.get() or exception in worker
- **Fix**: Check worker logs for exceptions, ensure result_queue.put() in all code paths

**Issue**: Loss not decreasing
- **Cause**: Learning signal too weak (low MCTS visits, buffer too small)
- **Fix**: Verify 16 visits enough, increase GAMES_PER_WORKER_PER_ROUND

**Issue**: Out of memory
- **Cause**: Batch size too large or buffer leaking
- **Fix**: Reduce BATCH_SIZE, check buffer max_size, monitor VRAM

**Issue**: Metrics show 0 games but buffer has data
- **Cause**: Counter key mismatch (variant name double-prefixed)
- **Fix**: Verify monitoring.py doesn't double-prefix variant names (FIXED in current version)

---

## Current Status

**Stage**: 2 (Intermediate Parallelism)
**Config**: 4 workers × 5 games × 16 visits (self-play), 64 visits (eval)
**Last Validated**: Round 0 (Stage 1 bootstrap, no timeouts)
**Next Milestone**: 10 rounds at Stage 2 with loss decreasing
