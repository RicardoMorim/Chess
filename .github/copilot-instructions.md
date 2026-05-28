---
name: "copilot-instructions"
description: "Short, always-on agent instructions for contributors and AI coding agents interacting with this Chess project. Keep minimal; prefer linking to AGENTS.md and CLAUDE.md."
applyTo:
  - "**/*.py"
  - "train/**"
  - "docs/**"
---

# Copilot instructions (minimal)

This repository is a chess AI system combining a playable UI and a training pipeline. For full guidance and architecture details, see `AGENTS.md` and `CLAUDE.md` at the repo root.

- Prefer running `python -m unittest discover` for tests. Use `test_parallel.py` for Minimax comparisons.
- Building the Cython extension is required after editing `.pyx` files: `python compile_cython.py`.
- Do not duplicate long documentation; link into `docs/` or `CLAUDE.md`.
