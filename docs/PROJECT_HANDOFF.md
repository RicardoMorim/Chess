# Project handoff — Chess AI repo

Last updated: 2026-05-29

This document is a compact but complete handoff for restarting work in a new session. It summarizes how the project works, what we were trying to do, what has already been done, and what should happen next.

## 1) What this project is

This repository is a chess AI system with three major parts:

- **Game/UI layer** — playable chess UI in Python + pygame.
- **Training layer** — neural-network training, curriculum learning, self-play, puzzle bootstrapping, and league training.
- **Engine layer** — a fast Cython-backed Minimax engine used by the playable side and benchmark tooling.

The project is not a single model or single trainer. It has **two training philosophies**:

- `train/individual`: guided curriculum training
- `train/league`: self-play / population-style training

These are complementary, not duplicates.

## 2) How the project works

### Gameplay / UI

Key entrypoints and files:

- `Main.py` / `main.py` — UI entrypoint
- `Minimax.py`, `Minimax_improved.py` — Python wrappers over the engine
- `minimax_cy.pyx`, `minimax_improved_cy.pyx` — Cython engine sources
- `compile_cython.py` — builds the extension

The UI can play against either a neural net or Minimax depending on settings.

### Core training primitives

Key files under `train/core`:

- `models.py` — `ChessNet`, `ESTNet`, and `create_model()`
- `training.py` — loss functions, optimizer/scheduler factories, legacy training helpers
- `mcts.py` — MCTS implementation used by self-play
- `data.py` — board tensors, move indexing, datasets, PGN/puzzle loaders
- `constants.py` — model config, training config, MCTS config, self-play config
- `repro.py` — centralized seed helpers (`set_seed`, `get_seed_from_env`)
- `lightning_module.py` — Lightning wrapper around the chess model

### Training data layout

The repo expects chess PGN data under:

- `train/chess_pgns/pros/` — pro / classical games
- `train/chess_pgns/high_elo/` — strong high-ELO games
- `train/chess_pgns/puzzles/` — tactical puzzles

The new recursive loaders in `train/core/data.py` can discover and load those folders.

## 3) Training modes and the difference between them

### `train/individual`

This is the **curriculum / supervised-first** pipeline.

Files:

- `train/individual/main.py`
- `train/individual/curriculum.py`
- `train/individual/checkmate.py`
- `train/individual/selfplay_parallel.py`

How it trains:

1. **Phase 1: Puzzle bootcamp**
   - trains on puzzles first
   - goal: learn tactical patterns and checkmates

2. **Phase 2: Transition**
   - blends self-play with supervised data
   - now also uses the local `pros/` and `high_elo/` PGNs

3. **Phase 3: Pure self-play**
   - long-running convergence loop
   - still part curriculum, but uses self-play heavily

Core idea:

- teach the model to understand chess from curated data first
- then use self-play to improve

### `train/league`

This is the **pure self-play / competition** pipeline.

Files:

- `train/league/main.py`
- `train/league/league_trainer.py`
- `train/league/self_play_worker.py`
- `train/league/replay_buffer.py`
- `train/league/evaluator.py`
- `train/league/monitoring.py`

How it trains:

1. Spawn self-play workers per variant (`baseline`, `attack`, `est`)
2. Generate games using MCTS
3. Store trajectories in replay buffers
4. Train each model from its replay buffer
5. Save checkpoints and evaluate progress

Core idea:

- models improve by playing themselves and competing against sibling variants
- replay buffer is the main training source

### Main difference

- `individual` = curriculum learning, puzzle bootstrapping, supervised blending
- `league` = population self-play, replay buffers, model-vs-model improvement

## 4) What we were trying to do recently

We were improving the AI training stack in the following direction:

- make the project easier to resume in a fresh session
- add Lightning as a cleaner trainer wrapper
- add Optuna hyperparameter tuning
- add experiment tracking with Weights & Biases
- make training use the repo’s real PGN folders (`pros`, `high_elo`, `puzzles`)
- keep self-play and model-vs-model training as first-class parts of the system

## 5) What has already been done

### Infrastructure and tooling

- Added project-level guidance files:
  - `AGENTS.md`
  - `.github/copilot-instructions.md`
- Added benchmark/repro tooling:
  - `tools/reproducible_benchmark.py`
  - `tools/smoke_import.py`
- Added Optuna HPO runner:
  - `tools/optuna_hpo.py`
  - `tools/hpo_example.json`
- Added a Lightning quickstart script:
  - `train/run_lightning_quickstart.py`
- Added CI smoke workflow:
  - `.github/workflows/ci-smoke.yml`

### Training / ML changes

- Centralized seeding in `train/core/repro.py`
- Added `train/core/lightning_module.py`
- Refactored Optuna objective to use Lightning trainer
- Added W&B support in the league metrics/training path
- Added recursive PGN loaders in `train/core/data.py`
- Updated `train/individual/curriculum.py` and `train/individual/main.py` to prefer local PGN data

### Tests added

- `tests/test_pgn_loaders.py`
- `tests/test_quickstart.py`
- `tests/__init__.py`

### Dependency updates

- `optuna`
- `pytorch-lightning`
- `wandb`
- `Cython`

## 6) Verification status

What has been checked locally:

- `python -m unittest discover -s tests -p "test_*.py" -v` ✅
- `python tools/smoke_import.py` ✅ after making imports package-safe

Notes:

- The quickstart test skips locally if `pytorch_lightning` is not usable in the current environment.
- There are still long-standing complexity warnings in older files (`train/core/data.py`, `train/individual/main.py`, `train/league/league_trainer.py`, etc.). They did not block the new work.

## 7) Important implementation details

### Self-play

The league system is the strongest self-play implementation in the repo.

- `self_play_worker.py` runs MCTS-guided games
- trajectories are stored as `(position, policy, value)` samples
- `league_trainer.py` manages the round loop:
  - self-play
  - training
  - checkpointing
  - buffer pruning
  - evaluation

### PGN data usage

The new loaders in `train/core/data.py` can recursively load:

- pro games
- high-ELO games
- puzzle PGNs

This matters because the project now has a real supervised data source instead of only a Lichess puzzle cache.

### W&B logging

W&B is optional and is enabled from the league metrics layer.

- env vars used in league:
  - `WANDB_PROJECT`
  - `WANDB_RUN_NAME`
  - `WANDB_MODE`

The run is initialized lazily and logs round summaries.

### Lightning

Lightning is used as a cleaner wrapper for the model training path.

- `ChessLightning` wraps the chess model
- The Optuna objective now trains via `Trainer`
- The quickstart script exists mainly as a CI smoke test and a minimal reproducible training run

## 8) Current project state

The repo is now in a better place than when we started:

- training can bootstrap from puzzles and supervised PGNs
- self-play remains the long-term improvement loop
- league training supports W&B-style monitoring
- tests exist for the new PGN loaders and quickstart flow
- CI smoke workflow validates the basics

## 9) Next steps

These are the remaining most useful tasks:

1. **Add / expand automated tests and coverage target**
   - current tests are basic smoke tests and PGN loader tests
   - coverage target still needs to be defined

2. **Add config management (Hydra) and a reproducible experiment runner**
   - would make training runs easier to reproduce and compare

3. **Refactor training loop into a pluggable Trainer**
   - one path for Lightning
   - one path for vanilla / legacy training

4. **Automate Cython build & packaging**
   - `pyproject.toml` / build metadata
   - wheel generation for the engine

5. **Improve docs for onboarding**
   - `CONTRIBUTING.md`
   - `DEV_SETUP.md`

6. **Add dependency hygiene**
   - linting
   - pre-commit
   - type checking
   - Dependabot / security scanning

7. **Make the league/individual split explicit in user docs**
   - when to use each
   - recommended training order

## 10) Recommended training order

If starting from scratch in a new session:

1. Install dependencies
2. Verify the PGN folders exist
3. Run `train/individual` on puzzles + supervised PGNs
4. Run `train/league` for self-play improvement
5. Use W&B / metrics / checkpoints to compare variants

## 11) Useful commands

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python tools/smoke_import.py
python train/run_lightning_quickstart.py --epochs 1 --batch-size 4 --seed 42
python train/individual/main.py --variant baseline
python league/main.py
python compile_cython.py
```

## 12) Caveats

- `train/core/__init__.py` had a bad import patch at one point, but it was fixed.
- Lightning imports can be fragile in this environment because of transitive dependency mismatches; the code now uses lazy/optional imports where practical.
- Many older complexity warnings still exist in the legacy training files. They are mostly technical debt, not current blockers.

## 13) If you resume in a new session

Start by checking:

- `docs/PROJECT_HANDOFF.md`
- `CLAUDE.md`
- `AGENTS.md`
- `train/individual/main.py`
- `train/league/main.py`

Then decide whether the next move is:

- improve tests/coverage,
- add Hydra/config management,
- or deepen the Lightning/league refactor.
