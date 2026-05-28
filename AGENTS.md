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

## Quick commands

- Install dependencies: `pip install -r requirements.txt`
- Build Cython extension: `python compile_cython.py` (produces compiled .pyd under `build/`)
- Run the game UI: `python Main.py` or `python main.py` depending on entrypoint
- Run training: `python train/train.py`
- Run tests: `python -m unittest discover` or `python test_parallel.py`

## Architecture & key files (links)

- Gameplay / UI: `Main.py`, `Minimax.py`, `Minimax_improved.py` — see `CLAUDE.md` and `Main.py`.
- Engine (Cython): `minimax_cy.pyx`, `minimax_improved_cy.pyx`, build scripts: `compile_cython.py`, `setup.py`.
- Training pipeline: `train/` (entry: `train/train.py`, models: `train/core/models.py`, mcts: `train/core/mcts.py`).
- Docs: `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`, `CLAUDE.md` (high-level guide).

## Common pitfalls

- Cython build requires matching Python ABI and may produce .pyd in `build/` — check `setup_cython.py` and `compile_cython.py` for platform-specific names.
- YAML/frontmatter and applyTo: keep instructions concise and prefer globs (e.g. `**/*.py`) — avoid `applyTo: "**"` unless truly global.
- When adding long documentation, link to `docs/` or `CLAUDE.md` rather than duplicating.

## Quick tips for agents

- When asked to run tests or reproduce a bug, prefer the `python -m unittest` command and point to `test_parallel.py` for Minimax checks.
- When editing Cython files, run the build script and run the Minimax tests to validate binary compatibility.

## Where to find more

- High-level developer guide: `CLAUDE.md`
- Technical docs: `docs/ARCHITECTURE.md`, `docs/QUICKSTART.md`
