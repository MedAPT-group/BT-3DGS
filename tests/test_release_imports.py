"""Public entrypoints must import without optional logging or a CUDA build."""

import os
from pathlib import Path
import subprocess
import sys


def test_entrypoints_import_without_wandb_or_cuda():
    code = """
import importlib
import sys
sys.modules['wandb'] = None
sys.modules['simple_knn'] = None
import torch
def forbidden(*args, **kwargs):
    raise AssertionError('Import attempted to access CUDA')
torch.cuda.is_available = forbidden
torch.cuda.init = forbidden
for name in ('train_binary', 'train_default', 'train_distill', 'train_lod', 'validate'):
    importlib.import_module('scripts.train.' + name)
from r2splat.cuda._backend import _lazy_C
assert not _lazy_C._load_attempted
assert 'r2splat.csrc' not in sys.modules
from src.loggers.wandb import WandbLogger
assert WandbLogger(None, '.').initialize() == (None, False)
"""
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1], env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
