"""CPU regressions for subtree export; CUDA, model I/O and rendering are stubbed."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf


@pytest.fixture
def renderer(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    gaussian, binary = Mock(), Mock()
    binary.return_value.num_gaussians = 2
    binary.return_value.history_means = torch.zeros(1, 3)
    tree = (torch.tensor([1, -1, -1]), torch.tensor([2, -1, -1]), torch.tensor([0, 1, 1]))
    for name, module in {
        "src.data.ctproj": SimpleNamespace(CTImageParser=Mock(), CTImageDataset=Mock()),
        "src.models.gaussian_model": SimpleNamespace(GaussianModel=gaussian),
        "src.models.binary_tree_gaussian_model": SimpleNamespace(BinaryTreeGaussianModel=binary),
        "src.models.student_gaussian_model": SimpleNamespace(StudentGaussianModel=SimpleNamespace(_build_tree=Mock(return_value=tree))),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    ns = runpy.run_path(str(root / "src/utils/seg_render.py"))
    return SimpleNamespace(ns=ns, gaussian=gaussian, binary=binary)


@pytest.fixture
def saved_run(tmp_path):
    run = tmp_path / "run"
    (run / "ckpts").mkdir(parents=True)
    checkpoint = run / "ckpts/ckpt_29999.pt"
    checkpoint.touch()
    cfg = OmegaConf.create({
        "volume_epsilon": 8e-9,
        "model": {
            "scale_min": 0.0005, "scale_max": 1.0, "scale_eps": 1e-5,
            "eps2d": 7e-8, "eps3d": "${volume_epsilon}",
            "img_tile_size": 32, "vol_tile_size": 4,
            "img_radius_clip": 2.0, "vol_radius_clip": 2.5, "backward_mode": "pixel",
        },
    })
    OmegaConf.save(cfg, run / "config_used.yaml")
    gt = tmp_path / "volume.npy"
    np.save(gt, np.zeros((3, 5, 1), dtype=np.float32))
    return SimpleNamespace(checkpoint=checkpoint, gt=gt)


@pytest.mark.parametrize("active_start", [1, 2])
def test_main_restores_config_for_teacher_full_and_subtree_models(renderer, saved_run, monkeypatch, tmp_path, active_start):
    main = renderer.ns["main"]
    ns = main.__globals__
    monkeypatch.setattr(sys, "argv", ["seg_render", "--ckpt", str(saved_run.checkpoint),
                                    "--data", str(tmp_path), "--out", str(tmp_path / "export"),
                                    "--max_depth", "1", "--view_indices", "0"])
    parser = SimpleNamespace(vol_gt_path=saved_run.gt, scanner={"nVoxel": [128, 128, 128]},
                             bbox=[[-1, -1, -1], [1, 1, 1]])
    monkeypatch.setitem(ns, "CTImageParser", Mock(return_value=parser))
    monkeypatch.setitem(ns, "CTImageDataset", Mock(return_value=[object()]))
    renderer.binary.return_value.num_gaussians = 3 - active_start
    renderer.binary.return_value.history_means = torch.zeros(active_start, 3)
    monkeypatch.setitem(ns, "merge_tree", Mock(return_value={
        "parents": torch.tensor([-1, 0, 0]), "active_start": active_start,
    }))
    subset = Mock(return_value=object())
    voxelize = Mock(return_value=np.zeros((3, 5, 1), dtype=np.float32))
    project = Mock(return_value=np.zeros((4, 6), dtype=np.float32))
    monkeypatch.setitem(ns, "build_subset_model", subset)
    monkeypatch.setitem(ns, "compute_volume", voxelize)
    monkeypatch.setitem(ns, "render_projection", project)
    main()
    kwargs = renderer.binary.call_args.kwargs.copy()
    assert kwargs.pop("device") == "cuda:0"
    assert kwargs == {
        "min_scale": 0.0005, "max_scale": 1.0, "scale_eps": 1e-5,
        "eps2d": 7e-8, "eps3d": 8e-9, "img_tile_size": 32, "vol_tile_size": 4,
        "img_radius_clip": 2.0, "vol_radius_clip": 2.5, "backward_mode": "pixel",
    }
    renderer.binary.return_value.load_ckpt.assert_called_once_with(str(saved_run.checkpoint))
    active = list(range(active_start, 3))
    subtree_groups = [active] + [[leaf] for leaf in active]
    # Check full volume, each subtree volume, and every projection use active IDs only.
    assert [list(call.args[0]) for call in subset.call_args_list] == [active] + subtree_groups * 2
    assert all(call.kwargs == kwargs for call in subset.call_args_list)
    assert voxelize.call_count == project.call_count == 2 + len(active)
    if active_start == 2:
        assert not (tmp_path / "export/depth_1/node_1").exists()
    assert all(call.args[1] == (3, 5, 1) for call in voxelize.call_args_list)
    assert np.load(tmp_path / "export/full_tree/volume.npy").shape == (3, 5, 1)


def test_missing_config_fails_before_loading_teacher(renderer, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["seg_render", "--ckpt", str(tmp_path / "missing.pt"),
                                    "--data", str(tmp_path), "--out", str(tmp_path / "export")])
    with pytest.raises(FileNotFoundError, match="No Hydra config_used.yaml"):
        renderer.ns["main"]()
    renderer.binary.assert_not_called()
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("device, available", [("cpu", True), ("cuda:0", False)])
def test_unsupported_device_fails_before_loading_models(renderer, monkeypatch, tmp_path, device, available):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(sys, "argv", ["seg_render", "--ckpt", "unused", "--data", "unused",
                                    "--out", str(tmp_path / "export"), "--device", device])
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        renderer.ns["main"]()
    renderer.binary.assert_not_called()
    assert not (tmp_path / "export").exists()


def test_subset_copies_raw_parameters_with_supplied_constructor_settings(renderer):
    fields = {"means": torch.arange(9, dtype=torch.float32).reshape(3, 3),
              "scales": torch.ones(3, 3), "quats": torch.ones(3, 4), "opacities": torch.ones(3)}
    kwargs = {"min_scale": 0.0005, "max_scale": 1.0, "eps2d": 7e-8, "eps3d": 8e-9}
    result = renderer.ns["build_subset_model"]([2, 1], {**fields, "active_start": 1}, torch.device("cpu"), **kwargs)
    renderer.gaussian.assert_called_once_with(device="cpu", **kwargs)
    for key, tensor in fields.items():
        value = getattr(result, key)
        torch.testing.assert_close(value, tensor[[2, 1]])
        assert not value.requires_grad


def test_exported_volume_matches_gt_axes_and_keeps_singleton_dimension(renderer):
    native = torch.arange(15, dtype=torch.float32).reshape(1, 5, 3, 1)
    model = SimpleNamespace(device=torch.device("cpu"), voxelize=Mock(return_value=native))
    result = renderer.ns["compute_volume"](model, (3, 5, 1), torch.zeros(2, 3))
    assert result.shape == (3, 5, 1)
    np.testing.assert_array_equal(result, native[..., 0].permute(2, 1, 0).numpy())
