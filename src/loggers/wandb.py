"""Utilities for initializing wandb runs used by trainers."""

from typing import Any, Optional, Tuple

try:
    import wandb
except ImportError:
    wandb = None


class WandbLogger:
    """Initialize wandb from trainer/runtime configuration."""

    def __init__(self, wandb_cfg: Any, result_dir: str):
        self.wandb_cfg = wandb_cfg
        self.result_dir = result_dir

    def initialize(self) -> Tuple[Optional[Any], bool]:
        """Return `(run, enabled)` after trying to initialize wandb."""
        cfg = self.wandb_cfg
        if cfg is None:
            return None, False
        if not bool(getattr(cfg, "enabled", False)):
            return None, False
        if wandb is None:
            print("⚠️ wandb is enabled but not installed, logging is disabled.")
            return None, False
        if wandb.run is not None:
            return wandb.run, True

        init_kwargs = {
            "project": getattr(cfg, "project", "bt-3dgs"),
            "dir": self.result_dir,
        }
        entity = getattr(cfg, "entity", None)
        run_name = getattr(cfg, "run_name", None)
        tags = getattr(cfg, "tags", None)
        if entity:
            init_kwargs["entity"] = entity
        if run_name:
            init_kwargs["name"] = run_name
        if tags:
            init_kwargs["tags"] = list(tags)

        try:
            run = wandb.init(**init_kwargs)
            return run, run is not None
        except Exception as exc:
            print(f"⚠️ Failed to initialize wandb ({type(exc).__name__}: {exc}), logging is disabled.")
            return None, False
