"""Randomness utilities controlled by config."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = True,
    cudnn_benchmark: bool = False,
    deterministic_warn_only: bool = True,
) -> None:
    """Seed python, numpy and torch RNGs with one config value."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(
            True, warn_only=bool(deterministic_warn_only)
        )
