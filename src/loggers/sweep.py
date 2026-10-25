"""Helpers for integrating wandb sweeps with Hydra configs."""

from __future__ import annotations

import os
from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf

try:
    import wandb
except ImportError:
    wandb = None


_SWEEP_PREFIXES = ("trainer", "optim", "strategy", "model", "exp", "random", "wandb")


def _build_init_kwargs(wandb_cfg: Any, result_dir: str) -> Dict[str, Any]:
    init_kwargs: Dict[str, Any] = {
        "project": getattr(wandb_cfg, "project", "bt-3dgs"),
        "dir": result_dir,
    }
    entity = getattr(wandb_cfg, "entity", None)
    run_name = getattr(wandb_cfg, "run_name", None)
    tags = getattr(wandb_cfg, "tags", None)
    if entity:
        init_kwargs["entity"] = entity
    if run_name:
        init_kwargs["name"] = run_name
    if tags:
        init_kwargs["tags"] = list(tags)
    return init_kwargs


def _try_apply_override(cfg: DictConfig, key: str, value: Any) -> bool:
    candidate_paths = [key]
    if "." not in key:
        candidate_paths.extend([f"{prefix}.{key}" for prefix in _SWEEP_PREFIXES])

    for path in candidate_paths:
        if OmegaConf.select(cfg, path) is None:
            continue
        OmegaConf.update(cfg, path, value, merge=False)
        return True
    return False


def _resolve_cfg_value(cfg: DictConfig, key: str) -> Tuple[Any, str]:
    candidate_paths = [key]
    if "." not in key:
        candidate_paths.extend([f"{prefix}.{key}" for prefix in _SWEEP_PREFIXES])
    for path in candidate_paths:
        value = OmegaConf.select(cfg, path)
        if value is not None:
            return value, path
    return None, ""


def _apply_bindings(cfg: DictConfig) -> None:
    """
    Apply deterministic relations between hyperparameters after sweep overrides.

    Expected config format:
      wandb:
        bindings:
          - source: optim.lr_means_init
            target: optim.lr_means_final
            op: mul
            factor: 0.1
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    bindings = getattr(wandb_cfg, "bindings", None) if wandb_cfg is not None else None
    if not bindings:
        return

    for item in bindings:
        source_key = getattr(item, "source", None)
        target_key = getattr(item, "target", None)
        op = str(getattr(item, "op", "mul")).lower()
        factor = float(getattr(item, "factor", 1.0))
        if not source_key or not target_key:
            continue

        source_value, _ = _resolve_cfg_value(cfg, str(source_key))
        if source_value is None:
            print(f"[wandb sweep] Binding skipped, source not found: {source_key}")
            continue

        source_value = float(source_value)
        if op == "mul":
            target_value = source_value * factor
        elif op == "div":
            target_value = source_value / factor
        elif op == "add":
            target_value = source_value + factor
        elif op == "sub":
            target_value = source_value - factor
        else:
            print(f"[wandb sweep] Binding skipped, unsupported op: {op}")
            continue

        if not _try_apply_override(cfg, str(target_key), target_value):
            print(f"[wandb sweep] Binding skipped, target not found: {target_key}")
            continue

        print(
            f"[wandb sweep] Binding applied: {target_key} = {source_key} {op} {factor} -> {target_value}"
        )


def apply_sweep_overrides(cfg: DictConfig, result_dir: str) -> Tuple[DictConfig, bool]:
    """
    Apply wandb sweep parameters into Hydra config before trainer construction.

    Returns:
        (cfg, sweep_active)
    """
    wandb_cfg = getattr(cfg, "wandb", None)
    if wandb_cfg is None or not bool(getattr(wandb_cfg, "enabled", False)):
        return cfg, False
    if wandb is None:
        return cfg, False

    sweep_id = os.getenv("WANDB_SWEEP_ID")
    if not sweep_id:
        return cfg, False

    run = wandb.run
    if run is None:
        run = wandb.init(**_build_init_kwargs(wandb_cfg=wandb_cfg, result_dir=result_dir))
    if run is None:
        return cfg, False

    sweep_cfg = dict(run.config)
    applied = []
    skipped = []
    for key, value in sweep_cfg.items():
        if _try_apply_override(cfg, key, value):
            applied.append(key)
        else:
            skipped.append(key)

    _apply_bindings(cfg)

    # Avoid output collisions across sweep runs when user keeps the same result_dir.
    if run.id:
        cfg.exp.result_dir = os.path.join(cfg.exp.result_dir, "sweeps", f"sweep_{run.id}")

    if applied:
        print(f"[wandb sweep] Applied overrides: {', '.join(applied)}")
    if skipped:
        print(f"[wandb sweep] Skipped unknown keys: {', '.join(skipped)}")

    return cfg, True
