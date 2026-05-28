"""Simple smoke test to import standalone modules we just added.

This bypasses package import paths by loading modules directly from file
locations so we avoid circular or relative import issues during quick checks.
"""
from pathlib import Path
import importlib.util
import sys

base = Path(__file__).resolve().parents[1] / 'train' / 'core'
files = ['repro.py', 'lightning_module.py']

for f in files:
    p = base / f
    spec = importlib.util.spec_from_file_location(f[:-3], str(p))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print('LOADED', f)
    except Exception as e:
        print('ERROR', f, e)
        sys.exit(2)

print('SMOKE_OK')
