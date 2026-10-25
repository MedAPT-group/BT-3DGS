"""Shared trainer abstractions for CT 3D Gaussian Splatting.

This module defines the lightweight data containers exchanged during
training and the abstract trainer interface implemented by concrete
trainer classes.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class TrainBatch:
    """Normalized training batch consumed by a trainer step.

    All tensor fields are expected to have already been moved onto the
    trainer device so downstream code can use them directly without
    additional device transfers.
    """

    raw: Dict[str, torch.Tensor]
    camtoworlds: torch.Tensor
    viewmats: torch.Tensor
    intrinsics: torch.Tensor
    target_images: torch.Tensor
    image_height: int
    image_width: int


@dataclass
class TrainStepOutput:
    """Artifacts produced by a single optimization step.

    This container keeps the forward-pass outputs and loss terms together
    so logging, visualization, and checkpointing code can consume a
    single object instead of many loosely related values.
    """

    batch: TrainBatch
    renders: torch.Tensor
    render_info: Dict[str, Any]
    volume_pred: Optional[torch.Tensor]
    loss_dict: Dict[str, torch.Tensor]


class BaseTrainer:
    """Abstract trainer interface shared by all trainer implementations.

    Subclasses are responsible for constructing training components,
    running the optimization loop, and exposing a logging hook with a
    stable signature.
    """

    def __init__(self, cfg, device: str):
        """Store the runtime configuration and target device.

        Args:
            cfg: Experiment configuration object used by the trainer.
            device: Torch device identifier such as ``"cuda"`` or ``"cpu"``.
        """
        self.cfg = cfg
        self.device = device

    def build_training(self) -> None:
        """Create all state required before entering the training loop.

        Typical responsibilities include initializing the model,
        optimizers, training strategy, and loss computation utilities.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def train(self) -> None:
        """Run the end-to-end optimization loop."""
        raise NotImplementedError("Subclasses must implement this method.")

    def log(self, step: int, loss_dict: Dict[str, torch.Tensor]) -> None:
        """Record training metrics for a finished optimization step.

        Args:
            step: Zero-based global training step.
            loss_dict: Mapping from loss name to scalar tensor produced at
                the current step.
        """
        raise NotImplementedError("Subclasses must implement this method.")


Trainer = BaseTrainer
