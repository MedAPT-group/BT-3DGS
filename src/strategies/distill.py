from __future__ import annotations

from typing import Any

from .default import DefaultStrategy


class DistillStrategy(DefaultStrategy):
    """Default densification strategy reused for student distillation."""

    def __init__(self, cfg: Any):
        super().__init__(cfg)
