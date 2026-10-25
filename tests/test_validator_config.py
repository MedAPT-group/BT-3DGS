"""Validation preserves the operator settings for nested LoD exports."""

import pytest
from omegaconf import OmegaConf

from src.utils.validate import Validator


@pytest.mark.parametrize("relative_path", [
    "ckpts/point_cloud_29999.pickle",
    "lod_1024/ckpts/point_cloud_29999.pickle",
])
def test_validation_recovers_nearest_saved_model_config(tmp_path, relative_path):
    run = tmp_path / "run"
    run.mkdir()
    OmegaConf.save(OmegaConf.create({"model": {"eps2d": 1e-8}}), tmp_path / "config_used.yaml")
    OmegaConf.save(OmegaConf.create({
        "epsilon": 4e-8,
        "model": {"eps2d": "${epsilon}", "scale_min": 0.0005, "scale_max": 1.0},
    }), run / "config_used.yaml")

    assert Validator._load_training_model_config(str(run / relative_path)) == {
        "eps2d": 4e-8, "scale_min": 0.0005, "scale_max": 1.0,
    }


def test_validation_allows_legacy_exports_without_saved_config(tmp_path):
    assert Validator._load_training_model_config(str(tmp_path / "model.pickle")) == {}
