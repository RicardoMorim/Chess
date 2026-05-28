"""
Repro utilities: centralized seeding and deterministic helpers.

Keep a single source of truth for deterministic settings so tools and
trainers can import and reuse it. This prevents subtle inconsistencies
between scripts and tests.
"""
from __future__ import annotations

import os
import random
import time
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int] = None, deterministic: bool = True) -> int:
    """Set global random seeds for reproducibility.

    Args:
        seed: If None a seed is taken from the environment variable
              ``TRAIN_SEED`` or generated randomly.
        deterministic: If True, enable deterministic cuDNN behavior.

    Returns:
        The seed that was applied.
    """
    if seed is None:
        env = os.environ.get('TRAIN_SEED')
        if env is not None:
            try:
                seed = int(env)
            except Exception:
                seed = int(time.time() * 1000) % (2 ** 31)
        else:
            seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Allow cuDNN autotuner for performance when determinism is not required
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return seed


def get_seed_from_env(default: int = 42) -> int:
    """Read seed from ``TRAIN_SEED`` environment variable if present."""
    v = os.environ.get('TRAIN_SEED')
    if v is None:
        return default
    try:
        return int(v)
    except Exception:
        return default
