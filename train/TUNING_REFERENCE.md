"""
QUICK TUNING REFERENCE
======================

✅ BOTH OPTIMIZATIONS ARE ENABLED BY DEFAULT - FULLY ACTIVE RIGHT NOW
   • Disk Management: ✓ Running automatically every 10 rounds
   • Adaptive MCTS: ✓ Running automatically every 5 rounds
   • You don't need to tune anything unless you see problems

Default values are production-tested on RTX 5080 + Intel Ultra 9 (24-core).
Adjust ONLY if you see specific problems (e.g., throughput < 8 games/min).
"""

# ============================================================================
# DISK MANAGEMENT (No tuning needed in 99% of cases)
# ============================================================================

# File: train/league/league_trainer.py, lines ~73–81

MAX_BUFFER_FILES_PER_VARIANT = 3            # Keep 3 recent = ~60 MB per variant
DISK_USAGE_CHECK_EVERY_N_ROUNDS = 10        # Check every 10 rounds (safe frequency)
CRITICAL_DISK_THRESHOLD_PCT = 5             # Alert when < 5% free


# ============================================================================
# ADAPTIVE MCTS VISITS (Tune if throughput target differs from default)
# ============================================================================

# File: train/league/league_trainer.py, lines ~101–108

TARGET_GAMES_PER_MINUTE = 10                # Default target for RTX 5080 + 6 workers
                                            # If actual ≈ 8: reduce TARGET to 8
                                            # If actual ≈ 15: increase TARGET to 15

ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS = 5    # Check every 5 rounds (moderate lag)
                                            # If too noisy: increase to 10
                                            # If too slow to adapt: decrease to 3

VISITS_ADJUSTMENT_FACTOR = 0.15             # 15% step size = stable convergence
                                            # If oscillating: reduce to 0.05 or 0.10
                                            # If slow to adapt: increase to 0.25

MIN_MCTS_VISITS = 6                         # Never go below 6 (conservative)
MAX_MCTS_VISITS = 32                        # Never go above 32 (aggressiveness limit)


# ============================================================================
# GPU BATCHING (Optional, advanced)
# ============================================================================

# File: train/league/gpu_inference_server.py, lines ~52–57 (if enabled)

# NOTE: Default is DISABLED. Set use_gpu_batching=True in LeagueTrainer.run()

BATCH_SIZE = 32                             # Target batch size
                                            # If GPU util < 70%: increase to 64 or 128
                                            # If memory OOM: reduce to 16

POST_BATCH_WAIT_MS = 10                     # Wait 10ms to accumulate batch
                                            # If latency high: reduce to 5
                                            # If GPU util low: increase to 20

MAX_QUEUE_SIZE = 1000                       # Prevent queue overflow
                                            # If workers bottleneck: increase to 2000


# ============================================================================
# DIAGNOSTIC: Check if tuning is needed
# ============================================================================

"""
SYMPTOM                          → LIKELY CAUSE           → TUNE THIS
─────────────────────────────────────────────────────────────────────
Games/min < 5                    → MCTS visits too high   → Reduce TARGET or MAX_MCTS_VISITS
Games/min > 20                   → MCTS visits too low    → Increase MIN_MCTS_VISITS
Disk filling up                  → Buffers accumulating   → Check MAX_BUFFER_FILES_PER_VARIANT
Adaptive visits oscillating ±3   → Adjustment too large   → Reduce VISITS_ADJUSTMENT_FACTOR
Throughput not converging        → Lag too long           → Reduce ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS
"""


# ============================================================================
# QUICK TUNING SCENARIOS
# ============================================================================

"""
SCENARIO 1: My GPU is slow (games/min < target)
───────────────────────────────────────────────
  trainer.TARGET_GAMES_PER_MINUTE = 8           # Lower target expectation
  trainer.MAX_MCTS_VISITS = 24                  # Don't search as hard
  trainer.run()

SCENARIO 2: My GPU is fast (games/min > target)
────────────────────────────────────────────────
  trainer.TARGET_GAMES_PER_MINUTE = 15          # Higher target
  trainer.VISITS_ADJUSTMENT_FACTOR = 0.25       # Larger steps to improve quality faster
  trainer.run()

SCENARIO 3: I have 100 GB free disk, not worried about space
──────────────────────────────────────────────────────────────
  trainer.MAX_BUFFER_FILES_PER_VARIANT = 10     # Keep more buffers for longer history
  trainer.CRITICAL_DISK_THRESHOLD_PCT = 2       # Only warn if < 2% (very paranoid)
  trainer.run()

SCENARIO 4: Adaptive visits is too noisy
──────────────────────────────────────────
  trainer.VISITS_ADJUSTMENT_FACTOR = 0.05       # Small steps = smooth convergence
  trainer.ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS = 10  # Check less often
  trainer.run()

SCENARIO 5: I want to enable GPU batching
──────────────────────────────────────────
  trainer = LeagueTrainer(use_gpu_batching=True)  # Requires GPU inference setup
  # Then implement _board_to_features() and _move_to_index() in gpu_inference_server.py
  trainer.run()
"""


# ============================================================================
# ✅ IMMEDIATE USAGE (BOTH OPTIMIZATIONS ENABLED & WORKING)
# ============================================================================

"""
**Right now, disk management and adaptive MCTS are fully enabled.**

Start training with zero configuration - everything runs automatically:

    from train.league.league_trainer import LeagueTrainer
    from train.core.models import BigChessModel
    
    trainer = LeagueTrainer()  # ✅ Both optimizations enabled
    trainer.initialize_models(BigChessModel)
    trainer.run()  # ✅ Disk checks + adaptive tuning run automatically
    
    Expected log output:
      Round 5: Adaptive MCTS: 9.8 games/min (target 10). Adjusting visits 12 → 11
      Round 10: Disk usage: 82.4% free (42.1 GB). Pruned old buffer file...
      Round 15: Adaptive MCTS: 10.3 games/min (target 10). Adjusting visits 11 → 11
    
    That's it! Both optimizations work automatically in the background.
"""


# ============================================================================
# RESTORE DEFAULTS (if you made changes)
# ============================================================================

"""
To restore factory defaults, reset these constants in league_trainer.py:

TARGET_GAMES_PER_MINUTE = 10
VISITS_ADJUSTMENT_FACTOR = 0.15
ADAPTIVE_VISITS_CHECK_EVERY_N_ROUNDS = 5
MIN_MCTS_VISITS = 6
MAX_MCTS_VISITS = 32
MAX_BUFFER_FILES_PER_VARIANT = 3
DISK_USAGE_CHECK_EVERY_N_ROUNDS = 10
CRITICAL_DISK_THRESHOLD_PCT = 5
"""
