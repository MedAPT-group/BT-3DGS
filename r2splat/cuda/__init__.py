"""Lazy access to the compiled CUDA extension."""

from ._wrapper import is_cuda_extension_available

__all__ = ["is_cuda_extension_available"]
