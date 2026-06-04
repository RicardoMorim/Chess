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


# ============================================================================
# PERFORMANCE PRESETS (low_memory / eco / balanced / boost) — Fase 1
# ============================================================================

# File: train/league/performance.py

"""
Switch modes at runtime (no restart):

  trainer.set_mode("boost")          # applies on next round
  trainer.set_mode("eco")
  trainer.set_mode("low_memory")     # for laptops <32GB RAM
  trainer.set_auto_mode(True)        # trainer picks based on CPU usage

Or via the HTTP control server:

  curl -X POST http://127.0.0.1:7860/api/mode -d '{"mode":"boost"}'
  curl -X POST http://127.0.0.1:7860/api/auto_mode -d '{"enabled":true}'
"""

# Memory budget (3 variants x 22-channel fp16 position+policy, fp32 value):
PRESETS = {
    "low_memory": {"BATCH_SIZE":   64, "NUM_SELF_PLAY_WORKERS":  2, "MCTS_VISITS_SELFPLAY":  50, "REPLAY_BUFFER_MAX_SIZE":  20_000, "STOCKFISH_BENCH_EVERY_N_ROUNDS": 200, "PUZZLE_BATCHES_PER_GAME_BATCH": 0, "buffer_gb": 0.6},
    "eco":        {"BATCH_SIZE":  128, "NUM_SELF_PLAY_WORKERS":  3, "MCTS_VISITS_SELFPLAY":  80, "REPLAY_BUFFER_MAX_SIZE":  30_000, "STOCKFISH_BENCH_EVERY_N_ROUNDS": 100, "PUZZLE_BATCHES_PER_GAME_BATCH": 0, "buffer_gb": 1.0},
    "balanced":   {"BATCH_SIZE":  256, "NUM_SELF_PLAY_WORKERS":  6, "MCTS_VISITS_SELFPLAY": 200, "REPLAY_BUFFER_MAX_SIZE": 100_000, "STOCKFISH_BENCH_EVERY_N_ROUNDS":  25, "PUZZLE_BATCHES_PER_GAME_BATCH": 1, "buffer_gb": 3.3},
    "boost":      {"BATCH_SIZE": 1024, "NUM_SELF_PLAY_WORKERS": 12, "MCTS_VISITS_SELFPLAY": 400, "REPLAY_BUFFER_MAX_SIZE": 300_000, "STOCKFISH_BENCH_EVERY_N_ROUNDS":  10, "PUZZLE_BATCHES_PER_GAME_BATCH": 2, "buffer_gb": 10.0},
}

# Selecting a mode live (deferred to next round boundary via set_max_size):
#   trainer.set_knob("REPLAY_BUFFER_MAX_SIZE", 20_000)
# ReplayBuffer is backed by pre-allocated flat numpy arrays (no per-element
# Python objects), so a live shrink via set_max_size() preserves the most
# recent N entries and reuses the same storage.


# ============================================================================
# HOT-SWAP KNOBS — Fase 0
# ============================================================================

"""
Most training knobs are changeable at runtime without restarting the
trainer. The trainer's internal RLock guarantees thread-safety; the
ControlServer and Tkinter dashboard both call set_knob() under the hood.

  trainer.set_knob("BATCH_SIZE", 512)            # one knob
  trainer.set_knobs({"BATCH_SIZE": 512,          # batch update
                     "TRAINING_STEPS_PER_ROUND": 400})
  trainer.list_hot_knobs()                        # what's tunable

Or via HTTP:

  curl -X POST http://127.0.0.1:7860/api/knobs -d '{"knobs":{"BATCH_SIZE":512}}'
"""

# Knobs applied IMMEDIATELY (next training step):
HOT_KNOBS_IMMEDIATE = [
    "BATCH_SIZE",                          # GPU batch
    "TRAINING_STEPS_PER_ROUND",            # gradient steps per round
    "PUZZLE_BATCHES_PER_GAME_BATCH",       # # puzzle batches to mix in
    "PROGAME_BATCHES_PER_GAME_BATCH",      # # pro-game batches
    "POLICY_LOSS_WEIGHT",                  # loss weights
    "VALUE_LOSS_WEIGHT",
    "GPU_INFER_BATCH_SIZE",                # GPU batcher size
]

# Knobs applied at next ROUND BOUNDARY (deferred, requires re-init):
HOT_KNOBS_DEFERRED = [
    "MCTS_VISITS_SELFPLAY",                # MCTS search budget
    "NUM_SELF_PLAY_WORKERS",               # parallel games
    "SELF_PLAY_VARIANT_PARALLELISM",       # variants per round
    "REPLAY_BUFFER_MAX_SIZE",              # per-variant buffer cap
    "MCTS_VISITS_EVAL",                    # evaluation game visits
    "GAMES_PER_WORKER_PER_ROUND",          # how many games per worker
    "CHECKPOINT_EVERY_N_ROUNDS",           # checkpoint cadence
    "EVAL_EVERY_N_ROUNDS",                 # eval cadence
    "BUFFER_SAVE_EVERY_N_ROUNDS",          # buffer save cadence
    "METRICS_EVERY_N_ROUNDS",              # metrics write cadence
    "DISK_USAGE_CHECK_EVERY_N_ROUNDS",     # disk-check cadence
    "MAX_BUFFER_FILES_PER_VARIANT",        # buffer file retention
    "TARGET_GAMES_PER_MINUTE",             # adaptive MCTS target
    "VISITS_ADJUSTMENT_FACTOR",            # adaptive MCTS step
    "STOCKFISH_BENCH_EVERY_N_ROUNDS",      # Stockfish benchmark cadence
    "STOCKFISH_BENCH_NUM_GAMES",           # # games per benchmark
    "STOCKFISH_BENCH_TIME_LIMIT_MS",       # time control per game
]

# NOT hot-settable (require restart or model rebuild):
NOT_HOT_SETTABLE = [
    "VARIANTS",                            # variant list
    "INITIAL_LR", "LR_MILESTONES", "LR_GAMMA",  # LR schedule
    "USE_PUZZLE_INJECTION", "USE_PROGAME_INJECTION",  # toggles need a reload
    "C_PUCT", "TEMPERATURE_*",             # search hyperparams
    "SELF_PLAY_DEVICE", "use_gpu_batching",       # device changes
]


# ============================================================================
# CONTROL SERVER + DASHBOARDS — Fase 2/3
# ============================================================================

"""
The trainer auto-starts a stdlib HTTP control server on
http://127.0.0.1:7860 (loopback only) when constructed. The browser
dashboard lives under /, and SSE events stream over /api/matches/stream.

Endpoints:
  GET  /                        - Browser dashboard (vanilla HTML+Chart.js)
  GET  /api/status              - Trainer state snapshot (JSON)
  GET  /api/checkpoints         - List checkpoint files with metadata
  GET  /api/variants            - List active variants + buffer fill
  GET  /api/modes               - List performance modes + describe each
  GET  /api/knobs               - List hot-settable knobs + current values
  GET  /api/matches             - List in-flight / recent matches
  GET  /api/matches/stream      - SSE stream of match events
  POST /api/mode                - Switch mode  {"mode": "boost"}
  POST /api/knobs               - Hot-swap     {"knobs": {"BATCH_SIZE": 512}}
  POST /api/auto_mode           - Toggle       {"enabled": true}
  POST /api/pause               - Pause/Resume {"paused": true}
  POST /api/matches             - Queue match  {"type":"model","params":{...}}

Run the Tkinter dashboard in a separate process:
  cd train && python -m league.dashboard_tk

Security: bound to 127.0.0.1 only by default. Set LeagueTrainer(control_host="0.0.0.0")
explicitly to expose on LAN (NOT recommended without auth).
"""


# ============================================================================
# SPECTATE + PUZZLE SIDECAR — Fase 4/4b
# ============================================================================

"""
The spectate worker drains a queue of model-vs-model and puzzle-drill
matches and publishes events via the control server's MatchEventBus
(SSE-backed). Two match types:

  type: "model"  params: {white, black, visits, start_fen?}
  type: "puzzle" params: {puzzle_id?, visits}

Model names: a live variant ("baseline", "attack", "est") or a
checkpoint spec ("baseline_step_35") to load a frozen .pt.

Puzzle drills need the sidecar (Fase 4b). The cached tensor files
don't preserve FENs, so we parse the original CSV once into a small
sidecar:

  cd train
  python -m league.puzzle_sidecar                 # builds train/cache/puzzles_meta.pkl
  # or, from the repo root:
  python train/build_puzzle_sidecar.py

  # Quick smoke (first 1000 puzzles only):
  python train/build_puzzle_sidecar.py --max-rows 1000

The sidecar is loaded lazily on the first drill request and kept in
memory (~300MB for the full Lichess DB).
"""

