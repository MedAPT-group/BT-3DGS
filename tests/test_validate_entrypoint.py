"""CPU-only entrypoint tests; rendering and CUDA dependencies are stubbed."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf


@pytest.fixture
def validation_entrypoint(monkeypatch):
    cuda_guard = Mock(return_value="cuda:0")
    seed = Mock()
    validator = Mock()
    validator_config = Mock()
    monkeypatch.setitem(
        sys.modules, "src.utils.validate",
        SimpleNamespace(Validator=validator, ValidatorConfig=validator_config),
    )
    monkeypatch.setitem(
        sys.modules, "src.utils.random", SimpleNamespace(seed_everything=seed)
    )
    runtime = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src/utils/runtime.py"))
    monkeypatch.setitem(
        sys.modules, "src.utils.runtime", SimpleNamespace(
            require_scene_dir=runtime["require_scene_dir"],
            require_checkpoint_file=runtime["require_checkpoint_file"],
            require_cuda_device=cuda_guard,
        ),
    )
    entrypoint = Path(__file__).resolve().parents[1] / "scripts/train/validate.py"
    namespace = runpy.run_path(str(entrypoint))
    return SimpleNamespace(
        main=namespace["main"].__wrapped__, seed=seed, validator=validator,
        validator_config=validator_config, cuda_guard=cuda_guard,
    )


@pytest.mark.parametrize(
    "invalid_input",
    ["missing-checkpoint", "checkpoint-is-directory", "missing-data", "data-is-file",
     "missing-metadata", "metadata-is-directory"],
)
def test_invalid_inputs_raise_before_validation(
    invalid_input, tmp_path, validation_entrypoint
):
    checkpoint = tmp_path / "checkpoint.pickle"
    data_dir = tmp_path / "data"
    if invalid_input == "checkpoint-is-directory":
        checkpoint.mkdir()
    elif invalid_input != "missing-checkpoint":
        checkpoint.touch()
    if invalid_input == "data-is-file":
        data_dir.touch()
    elif invalid_input != "missing-data":
        data_dir.mkdir()
    if invalid_input == "metadata-is-directory":
        (data_dir / "meta_data.json").mkdir()
    result_dir = tmp_path / "results"
    cfg = OmegaConf.create({
        "checkpoint_path": str(checkpoint),
        "data_dir": str(data_dir),
        "result_dir": str(result_dir),
    })
    if "checkpoint" in invalid_input:
        expected = "Checkpoint file not found"
    elif "metadata" in invalid_input:
        expected = "Scene metadata not found"
    else:
        expected = "Data directory not found"

    with pytest.raises(FileNotFoundError, match=expected) as caught:
        validation_entrypoint.main(cfg)

    missing_path = checkpoint if "checkpoint" in invalid_input else data_dir
    assert str(missing_path) in str(caught.value)
    validation_entrypoint.seed.assert_not_called()
    validation_entrypoint.cuda_guard.assert_not_called()
    validation_entrypoint.validator.assert_not_called()
    assert not result_dir.exists()


def test_valid_inputs_reach_validator(tmp_path, validation_entrypoint):
    checkpoint = tmp_path / "checkpoint.pickle"
    checkpoint.touch()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "meta_data.json").touch()
    cfg = OmegaConf.create({
        "checkpoint_path": str(checkpoint), "data_dir": str(data_dir), "result_dir": None,
        "device": "cuda:0", "eval_save_images_n": 0, "ssim_window": 11,
        "num_workers": 0, "pin_memory": False,
        "scale_min": None, "scale_max": None, "scale_eps": 1e-5,
        "random": {"seed": 42, "deterministic": True, "cudnn_benchmark": False,
                   "deterministic_warn_only": True},
    })

    result = validation_entrypoint.main(cfg)

    assert result is validation_entrypoint.validator.return_value.validate.return_value
    validation_entrypoint.seed.assert_called_once()
    validation_entrypoint.cuda_guard.assert_called_once_with("cuda:0")
    validation_entrypoint.validator.assert_called_once_with(
        validation_entrypoint.validator_config.return_value, "cuda:0"
    )
    validation_entrypoint.validator.return_value.validate.assert_called_once_with()
    assert validation_entrypoint.validator_config.call_args.kwargs["checkpoint_path"] == str(checkpoint)
    assert validation_entrypoint.validator_config.call_args.kwargs["data_dir"] == str(data_dir)


def test_cuda_failure_precedes_validation_side_effects(tmp_path, validation_entrypoint):
    checkpoint = tmp_path / "checkpoint.pickle"
    checkpoint.touch()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "meta_data.json").touch()
    result_dir = tmp_path / "results"
    cfg = OmegaConf.create({
        "checkpoint_path": str(checkpoint), "data_dir": str(data_dir),
        "result_dir": str(result_dir), "device": "cuda:0",
    })
    validation_entrypoint.cuda_guard.side_effect = RuntimeError("CUDA unavailable")

    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        validation_entrypoint.main(cfg)

    validation_entrypoint.seed.assert_not_called()
    validation_entrypoint.validator.assert_not_called()
    assert not result_dir.exists()
