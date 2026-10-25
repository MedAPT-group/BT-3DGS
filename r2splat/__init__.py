"""Gaussian-only CT splatting operators."""

__version__ = "0.1.0"

from .ops import ExplicitGaussianState, InferenceGaussianState

__all__ = ["__version__", "ExplicitGaussianState", "InferenceGaussianState"]
