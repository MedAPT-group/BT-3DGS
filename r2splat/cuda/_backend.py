"""Lazy loader for the optional ``r2splat.csrc`` CUDA extension."""

from __future__ import annotations

import importlib

import torch


class _LazyModule:
    def __init__(self) -> None:
        self._module = None
        self._load_attempted = False

    def _try_load(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not torch.cuda.is_available():
            return
        try:
            self._module = importlib.import_module("r2splat.csrc")
        except ImportError:
            self._module = None

    def __getattr__(self, name: str):
        self._try_load()
        if self._module is None:
            raise RuntimeError(
                "r2splat CUDA extension is unavailable. Reinstall with "
                "`pip install -e . --no-build-isolation` after changing CUDA sources."
            )
        return getattr(self._module, name)

    def is_available(self) -> bool:
        self._try_load()
        return self._module is not None


_lazy_C = _LazyModule()


class _CProxy:
    def __getattr__(self, name: str):
        return getattr(_lazy_C, name)

    def __bool__(self) -> bool:
        return _lazy_C.is_available()


_C = _CProxy()


def is_cuda_available() -> bool:
    return _lazy_C.is_available()


__all__ = ["_C", "is_cuda_available"]
