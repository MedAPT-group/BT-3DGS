from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from omegaconf.errors import MissingMandatoryValue


@pytest.mark.parametrize("config_name", ["default", "binary_tree", "distill", "lod"])
def test_training_config_composes_with_new_operator_options(config_name):
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    assert cfg.model.img_tile_size == 16
    assert cfg.model.vol_tile_size == 8
    assert cfg.model.img_radius_clip == 3.0
    assert cfg.model.vol_radius_clip == 3.0
    assert cfg.model.backward_mode == "primitive"


@pytest.mark.parametrize("config_name", ["default", "binary_tree", "distill", "lod"])
def test_training_requires_explicit_scene_and_logging_is_opt_in(config_name):
    config_dir = str(Path(__file__).parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=config_name)
    with pytest.raises(MissingMandatoryValue):
        _ = cfg.exp.data_dir
    assert not cfg.wandb.enabled
    assert cfg.wandb.entity is None
    if config_name in {"distill", "lod"}:
        with pytest.raises(MissingMandatoryValue):
            _ = cfg.trainer.teacher_path


@pytest.mark.parametrize("config_name", ["default", "binary_tree", "distill", "lod", "validate"])
def test_documented_overrides_resolve_all_required_values(config_name):
    overrides = ["exp.data_dir=/path/to/scene", "exp.result_dir=output/example"]
    if config_name in {"distill", "lod"}:
        overrides.append("trainer.teacher_path=output/teacher/ckpts/ckpt_29999.pt")
    if config_name == "validate":
        overrides = ["data_dir=/path/to/scene", "checkpoint_path=output/example/ckpts/point_cloud_29999.pickle"]
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / "configs")):
        cfg = compose(config_name=config_name, overrides=overrides)
    assert OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
