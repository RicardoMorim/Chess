Tools: reproducible benchmark & reproducible training helper

Usage examples:

```bash
python tools/reproducible_benchmark.py --mode benchmark --iterations 3 --max-depth 3
python tools/reproducible_benchmark.py --mode repro-train --seed 42
```

## Hyperparameter optimization

Optuna HPO helper:

```bash
python tools/optuna_hpo.py --config tools/hpo_example.json
```

Quick overrides without a config file:

```bash
python tools/optuna_hpo.py --trials 5 --epochs 2 --seed 123
```

Notes:
- `repro-train` invokes `train.individual.main` with a tiny configuration (Phase 1 only) and sets global RNG seeds for `random`, `numpy`, and `torch` (if available).
- Ensure dependencies in `requirements.txt` are installed before running `repro-train`.
- The script is intentionally conservative (uses `--selfplay-workers 1`) to run on developer machines.
- The Optuna runner uses a compact tactical dataset built from `train/core/constants.py`, so it behaves like a smoke test for model/training changes rather than a full production sweep.
