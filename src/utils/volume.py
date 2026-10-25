"""Lightweight helpers for the scene volume grid."""

from pathlib import Path
from typing import Any

import numpy as np


def volume_shape_from_parser(parser: Any) -> tuple[int, int, int]:
    """Use the stored GT grid (header only), or explicit scanner dimensions."""
    volume_path = Path(parser.vol_gt_path)
    if volume_path.is_file():
        volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
        shape = volume.shape
    else:
        shape = parser.scanner.get("nVoxel")
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3
            or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v <= 0 for v in shape)):
        raise ValueError(f"Expected a 3-D GT volume or scanner.nVoxel with three positive integers: {volume_path}")
    return tuple(int(v) for v in shape)
