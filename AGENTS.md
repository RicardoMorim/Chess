---
name: AGENTS.md
description: "Project-level agent instructions to help AI coding agents work productively with this Chess repo. Links to detailed docs; quick commands; common pitfalls."
---

# Agent guidance for this repository

Keep this file minimal. Link to detailed docs rather than copying them.

## Use when

- Use when you need to run or modify the game UI (`Main.py`, `Minimax.py`, `Minimax_improved.py`).
- Use when you need to build or debug the Cython Minimax extension (`minimax_cy.pyx`, `minimax_improved_cy.pyx`).
- Use when you need to run training or training helpers under `train/`.
- Use when writing or running tests (`test_parallel.py`).
- Use when you need to operate the live control plane (HTTP/dashboards/spectate) under `train/league/`.

## Quick commands

- Install dependencies: `pip install -r requirements.txt`
- Build Cython extension: `python compile_cython.py` (produces compiled .pyd under `build/`)
- Run the game UI: `python Main.py` or `python main.py` depending on entrypoint
- Run training: `python train/train.py` (auto-starts HTTP control on `127.0.0.1:7860`)
- Run Tkinter dashboard: `cd train && python -m league.dashboard_tk`
- Open browser dashboard: http://127.0.0.1:7860/
- Build puzzle sidecar (one-time, for spectate drills): `python train/build_puzzle_sidecar.py`
- Run tests: `python -m unittest discover`

## Architecture & key files (links)

- Gameplay / UI: `Main.py`, `Minimax.py`, `Minimax_improved.py` — see `CLAUDE.md` and `Main.py`.
- Engine (Cython): `minimax_cy.pyx`, `minimax_improved_cy.pyx`, build scripts: `compile_cython.py`, `setup.py`.
- Training pipeline: `train/` (entry: `train/train.py`, models: `train/core/models.py`, mcts: `train/core/mcts.py`).
- League / control plane (Fase 0-4): `train/league/` — `league_trainer.py`, `performance.py`, `control_server.py`, `spectate.py`, `puzzle_sidecar.py`, `dashboard/`, `dashboard_tk.py`.
- Docs: `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`, `train/TUNING_REFERENCE.md` (hot-swap knobs + presets), `CLAUDE.md` (high-level guide).

## Common pitfalls

- Cython build requires matching Python ABI and may produce .pyd in `build/` — check `setup_cython.py` and `compile_cython.py` for platform-specific names.
- YAML/frontmatter and applyTo: keep instructions concise and prefer globs (e.g. `**/*.py`) — avoid `applyTo: "**"` unless truly global.
- When adding long documentation, link to `docs/` or `CLAUDE.md` rather than duplicating.
- Hot-swap knobs: many are immediate (next training step), some are deferred (next round). See `train/TUNING_REFERENCE.md` for the split.
- Puzzle spectate drills need `train/cache/puzzles_meta.pkl` — build it once with `python train/build_puzzle_sidecar.py`.
- The HTTP control server binds to `127.0.0.1` only by default; do not change `control_host` without adding auth.
- **Memory budget:** 3 variants × buffer_size × 11KB. `balanced` = 3.3 GB; use `low_memory` (~0.6 GB) on laptops <32GB RAM.
- **ReplayBuffer is compact:** pre-allocated fp16 arrays (no per-element Python objects). Live `set_max_size()` keeps the most recent entries.
- **Performance presets (4 total):** `low_memory` (20K buf, 2 workers, 50 visits) → `eco` (30K, 3, 80) → `balanced` (100K, 6, 200) → `boost` (300K, 12, 400). Auto-mode walks this ladder.

## Quick tips for agents

- When asked to run tests or reproduce a bug, prefer the `python -m unittest` command and point to `test_parallel.py` for Minimax checks.
- When editing Cython files, run the build script and run the Minimax tests to validate binary compatibility.
- When asked to tune the trainer at runtime, prefer `trainer.set_knob()` over editing constants — changes are immediate or deferred safely.
- When asked to spectate, queue a match via `POST /api/matches` rather than building a worker manually.

## Where to find more

- High-level developer guide: `CLAUDE.md`
- Technical docs: `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`
- Trainer tuning: `train/TUNING_REFERENCE.md`
- HTTP control plane: `train/league/control_server.py` (URL_PATTERNS at top)
