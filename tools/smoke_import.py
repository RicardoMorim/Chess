"""Simple smoke test to import the training modules we just added.

Imports through the package namespace so relative imports resolve correctly.
"""
from pathlib import Path
import importlib
import sys

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

modules = ["train.core.repro", "train.core.lightning_module"]

for name in modules:
    try:
        importlib.import_module(name)
        print("LOADED", name)
    except Exception as e:
        print("ERROR", name, e)
        sys.exit(2)

print("SMOKE_OK")
