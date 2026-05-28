Tools: reproducible benchmark & reproducible training helper

Usage examples:

```bash
python tools/reproducible_benchmark.py --mode benchmark --iterations 3 --max-depth 3
python tools/reproducible_benchmark.py --mode repro-train --seed 42
```

Notes:
- `repro-train` invokes `train.individual.main` with a tiny configuration (Phase 1 only) and sets global RNG seeds for `random`, `numpy`, and `torch` (if available).
- Ensure dependencies in `requirements.txt` are installed before running `repro-train`.
- The script is intentionally conservative (uses `--selfplay-workers 1`) to run on developer machines.
