"""Validation entrypoint script."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from src.utils.validate import Validator, ValidatorConfig
from src.utils.random import seed_everything
from src.utils.runtime import require_checkpoint_file, require_cuda_device, require_scene_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="validate")
def main(cfg: DictConfig):
    data_dir = to_absolute_path(cfg.data_dir)
    checkpoint_path = to_absolute_path(cfg.checkpoint_path)
    result_dir = to_absolute_path(cfg.result_dir) if cfg.result_dir else None

    require_checkpoint_file(checkpoint_path)
    require_scene_dir(data_dir)
    device = require_cuda_device(cfg.device)

    seed_everything(
        seed=cfg.random.seed,
        deterministic=cfg.random.deterministic,
        cudnn_benchmark=cfg.random.cudnn_benchmark,
        deterministic_warn_only=cfg.random.deterministic_warn_only,
    )
    validator_cfg = ValidatorConfig(
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        result_dir=result_dir,
        eval_save_images_n=int(cfg.eval_save_images_n),
        ssim_window=int(cfg.ssim_window),
        num_workers=int(cfg.num_workers),
        pin_memory=bool(cfg.pin_memory),
        scale_min=cfg.scale_min,
        scale_max=cfg.scale_max,
        scale_eps=float(cfg.scale_eps),
        seed=int(cfg.random.seed),
    )
    validator = Validator(validator_cfg, device)
    report = validator.validate()
    return report


if __name__ == "__main__":
    main()
