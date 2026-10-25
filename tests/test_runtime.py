"""CPU-only preflight tests; CUDA and trainer calls are replaced with stubs."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runtime():
    # Read the helper without importing the training package or CUDA backend.
    return runpy.run_path(str(REPO_ROOT / "src/utils/runtime.py"))


@pytest.fixture
def fake_cuda(monkeypatch):
    cuda = SimpleNamespace(is_available=Mock(return_value=True), device_count=Mock(return_value=2))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    return cuda


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:-1", "cuda:abc", "cuda:0:1", None])
def test_non_cuda_devices_fail_without_querying_cuda(device, runtime, fake_cuda):
    with pytest.raises(ValueError, match="CUDA device is required"):
        runtime["require_cuda_device"](device)
    fake_cuda.is_available.assert_not_called()


def test_missing_cuda_has_no_cpu_fallback(runtime, fake_cuda):
    fake_cuda.is_available.return_value = False
    with pytest.raises(RuntimeError, match="CUDA is required"):
        runtime["require_cuda_device"]("cuda:0")
    fake_cuda.device_count.assert_not_called()


def test_unavailable_cuda_index_is_rejected(runtime, fake_cuda):
    with pytest.raises(ValueError, match="CUDA device is not visible"):
        runtime["require_cuda_device"]("cuda:2")


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_visible_cuda_device_is_preserved(device, runtime, fake_cuda):
    assert runtime["require_cuda_device"](device) == device


@pytest.mark.parametrize("profile,trainer_name", [
    ("binary", "BinaryTreeTrainer"), ("default", "DefaultTrainer"),
    ("distill", "DistillTrainer"), ("lod", "LoDTrainer"),
])
@pytest.mark.parametrize("failure", ["metadata", "cuda"])
def test_training_preflight_precedes_outputs_logging_and_seed(
    profile, trainer_name, failure, tmp_path, monkeypatch, runtime
):
    data_dir = tmp_path / "scene"
    data_dir.mkdir()
    if failure != "metadata":
        (data_dir / "meta_data.json").touch()
    teacher_path = tmp_path / "teacher.pt"
    teacher_path.touch()
    result_dir = tmp_path / "results"
    cfg = OmegaConf.create({
        "exp": {"data_dir": str(data_dir), "result_dir": str(result_dir)},
        "trainer": {"device": "cuda:0", "teacher_path": str(teacher_path)},
    })
    cuda_guard = Mock(side_effect=RuntimeError("CUDA unavailable"))
    seed, sweep, trainer = Mock(), Mock(), Mock()
    monkeypatch.setitem(sys.modules, "src.utils.runtime", SimpleNamespace(
        require_scene_dir=runtime["require_scene_dir"],
        require_checkpoint_file=runtime["require_checkpoint_file"],
        require_cuda_device=cuda_guard,
    ))
    monkeypatch.setitem(sys.modules, "src.utils.random", SimpleNamespace(seed_everything=seed))
    monkeypatch.setitem(sys.modules, "src.loggers.sweep", SimpleNamespace(apply_sweep_overrides=sweep))
    trainer_module = "binary_tree" if profile == "binary" else profile
    monkeypatch.setitem(sys.modules, f"src.trainers.{trainer_module}", SimpleNamespace(**{trainer_name: trainer}))
    entrypoint = runpy.run_path(str(REPO_ROOT / f"scripts/train/train_{profile}.py"))

    expected_error = FileNotFoundError if failure == "metadata" else RuntimeError
    with pytest.raises(expected_error):
        entrypoint["main"].__wrapped__(cfg)

    if failure == "metadata":
        cuda_guard.assert_not_called()
    seed.assert_not_called()
    sweep.assert_not_called()
    trainer.assert_not_called()
    assert not result_dir.exists()


@pytest.mark.parametrize("profile,trainer_name", [("distill", "DistillTrainer"), ("lod", "LoDTrainer")])
def test_missing_teacher_fails_before_cuda(profile, trainer_name, tmp_path, monkeypatch, runtime):
    data_dir = tmp_path / "scene"
    data_dir.mkdir()
    (data_dir / "meta_data.json").touch()
    result_dir = tmp_path / "results"
    cfg = OmegaConf.create({
        "exp": {"data_dir": str(data_dir), "result_dir": str(result_dir)},
        "trainer": {"device": "cuda:0", "teacher_path": str(tmp_path / "missing.pt")},
    })
    cuda_guard, seed, sweep, trainer = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setitem(sys.modules, "src.utils.runtime", SimpleNamespace(
        require_scene_dir=runtime["require_scene_dir"],
        require_checkpoint_file=runtime["require_checkpoint_file"],
        require_cuda_device=cuda_guard,
    ))
    monkeypatch.setitem(sys.modules, "src.utils.random", SimpleNamespace(seed_everything=seed))
    monkeypatch.setitem(sys.modules, "src.loggers.sweep", SimpleNamespace(apply_sweep_overrides=sweep))
    monkeypatch.setitem(sys.modules, f"src.trainers.{profile}", SimpleNamespace(**{trainer_name: trainer}))
    entrypoint = runpy.run_path(str(REPO_ROOT / f"scripts/train/train_{profile}.py"))

    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        entrypoint["main"].__wrapped__(cfg)

    cuda_guard.assert_not_called()
    seed.assert_not_called()
    sweep.assert_not_called()
    trainer.assert_not_called()
    assert not result_dir.exists()
