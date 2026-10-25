"""
Entry script for training.
"""
import os
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.loggers.sweep import apply_sweep_overrides
from src.trainers.default import DefaultTrainer
from src.utils.random import seed_everything
from src.utils.runtime import require_cuda_device, require_scene_dir


@hydra.main(version_base=None, config_path="../../configs", config_name="default")
def main(cfg: DictConfig):
    cfg.exp.data_dir = to_absolute_path(cfg.exp.data_dir)
    cfg.exp.result_dir = to_absolute_path(cfg.exp.result_dir)
    require_scene_dir(cfg.exp.data_dir)
    device = require_cuda_device(cfg.trainer.device)
    cfg, sweep_active = apply_sweep_overrides(cfg=cfg, result_dir=cfg.exp.result_dir)
    if sweep_active:
        cfg.exp.data_dir = to_absolute_path(cfg.exp.data_dir)
        require_scene_dir(cfg.exp.data_dir)
        device = require_cuda_device(cfg.trainer.device)
    cfg.exp.result_dir = to_absolute_path(cfg.exp.result_dir)

    seed_everything(
        seed=cfg.random.seed,
        deterministic=cfg.random.deterministic,
        cudnn_benchmark=cfg.random.cudnn_benchmark,
        deterministic_warn_only=cfg.random.deterministic_warn_only,
    )
    os.makedirs(cfg.exp.result_dir, exist_ok=True)
    config_save_path = os.path.join(cfg.exp.result_dir, "config_used.yaml")
    OmegaConf.save(cfg, config_save_path)
    trainer = DefaultTrainer(cfg, device)
    trainer.train()


if __name__ == "__main__":
    main()
