"""Reproducibility utilities."""

from __future__ import annotations

import os
import random

import numpy as np

try:
    import torch
except ImportError:  # Data preparation can run in a lightweight Python environment.
    torch = None


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
