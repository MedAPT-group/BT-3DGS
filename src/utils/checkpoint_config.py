"""Read inference settings from the Hydra configuration saved with a checkpoint."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_checkpoint_config(ckpt_path: str) -> tuple[Path, DictConfig]:
    checkpoint = Path(ckpt_path).expanduser().absolute()
    config_path = next(
        (parent / "config_used.yaml" for parent in checkpoint.parents
         if (parent / "config_used.yaml").is_file()),
        None,
    )
    if config_path is None:
        raise FileNotFoundError(
            f"No Hydra config_used.yaml found for checkpoint {checkpoint}. "
            "Keep the checkpoint inside its original training run directory."
        )
    saved_cfg = OmegaConf.load(config_path)
    if not OmegaConf.is_dict(saved_cfg):
        raise ValueError(f"Expected a configuration mapping in {config_path}")
    return config_path, saved_cfg


def model_kwargs_from_config(saved_cfg: DictConfig, config_path: Path) -> dict:
    model_cfg = saved_cfg.get("model")
    if not OmegaConf.is_dict(model_cfg):
        raise ValueError(f"Missing or invalid model configuration in {config_path}")
    model_cfg = OmegaConf.to_container(model_cfg, resolve=True, throw_on_missing=True)
    config_keys = {
        "scale_min": "min_scale",
        "scale_max": "max_scale",
        "scale_eps": "scale_eps",
        "eps2d": "eps2d",
        "eps3d": "eps3d",
        "img_tile_size": "img_tile_size",
        "vol_tile_size": "vol_tile_size",
        "img_radius_clip": "img_radius_clip",
        "vol_radius_clip": "vol_radius_clip",
        "backward_mode": "backward_mode",
    }
    missing = [key for key in config_keys if key not in model_cfg]
    if missing:
        raise ValueError(f"Missing model settings in {config_path}: {', '.join(missing)}")
    if (model_cfg["scale_min"] is None) != (model_cfg["scale_max"] is None):
        raise ValueError(f"scale_min and scale_max must both be set or null in {config_path}")
    return {argument: model_cfg[key] for key, argument in config_keys.items()}


def load_model_kwargs(ckpt_path: str) -> dict:
    config_path, saved_cfg = load_checkpoint_config(ckpt_path)
    kwargs = model_kwargs_from_config(saved_cfg, config_path)
    print(f"  Model config loaded from: {config_path}")
    return kwargs
