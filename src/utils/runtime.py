"""Early checks for the CUDA training and validation entrypoints."""

import os
import re


def require_scene_dir(data_dir: str) -> None:
    """Check scene paths without loading projections or volume data."""
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    metadata_path = os.path.join(data_dir, "meta_data.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"Scene metadata not found: {metadata_path}")


def require_checkpoint_file(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint file not found: {path}")


def require_cuda_device(device: str) -> str:
    """Reject unsupported devices before creating run outputs or loading data."""
    if not isinstance(device, str) or re.fullmatch(r"cuda(?::[0-9]+)?", device) is None:
        raise ValueError("A CUDA device is required; use 'cuda' or 'cuda:<index>'.")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for BT-3DGS training and rendering. "
            "Check the NVIDIA driver and CUDA-enabled PyTorch installation."
        )
    if ":" in device and int(device.split(":", 1)[1]) >= torch.cuda.device_count():
        raise ValueError(f"CUDA device is not visible: {device}")
    return device
