"""CPU-only checks for LoD checkpoint configuration; model I/O is stubbed."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import OmegaConf


@pytest.fixture
def lod_module(monkeypatch):
    gaussian = Mock()
    gaussian.return_value._get_scale_bound.return_value = None
    binary_tree = Mock()
    fields = {}
    for name in ("means", "scales", "quats", "opacities"):
        tensor = Mock(name=name)
        tensor.detach.return_value = tensor
        tensor.to.return_value = tensor
        tensor.clone.return_value = tensor
        fields[name] = tensor
    torch_load = Mock(return_value={"model": fields})
    fake_torch = SimpleNamespace(
        no_grad=lambda: lambda fn: fn,
        load=torch_load,
        nn=SimpleNamespace(Parameter=lambda tensor: tensor),
    )
    pyplot = SimpleNamespace()
    replacements = {
        "torch": fake_torch,
        "matplotlib": SimpleNamespace(pyplot=pyplot),
        "matplotlib.pyplot": pyplot,
        "src.data.ctproj": SimpleNamespace(CTImageDataset=Mock(), CTImageParser=Mock()),
        "src.models.gaussian_model": SimpleNamespace(GaussianModel=gaussian),
        "src.models.binary_tree_gaussian_model": SimpleNamespace(BinaryTreeGaussianModel=binary_tree),
        "src.utils.metric": SimpleNamespace(metric_proj=Mock(), metric_vol=Mock()),
    }
    for name, module in replacements.items():
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    namespace = runpy.run_path(str(root / "src/utils/lod_render.py"))
    return SimpleNamespace(
        namespace=namespace, gaussian=gaussian, binary_tree=binary_tree,
        torch_load=torch_load, fields=fields,
    )


@pytest.fixture
def saved_config(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = OmegaConf.create({
        "volume_epsilon": 8e-9,
        "model": {
            "scale_min": 0.0005, "scale_max": 1.0, "scale_eps": 1e-5,
            "eps2d": 7e-8, "eps3d": "${volume_epsilon}",
            "img_tile_size": 32, "vol_tile_size": 4,
            "img_radius_clip": 2.0, "vol_radius_clip": 2.5,
            "backward_mode": "pixel",
        },
    })
    OmegaConf.save(cfg, run_dir / "config_used.yaml")
    return run_dir, cfg


@pytest.mark.parametrize("relative_path", [
    "ckpts/ckpt_29999.pt",
    "lod/ckpt_lod_1024_step_100.pt",
    "lod_1024/ckpts/point_cloud_29999.pickle",
])
def test_checkpoint_layouts_restore_saved_hydra_parameters(lod_module, saved_config, relative_path):
    run_dir, _ = saved_config
    kwargs = lod_module.namespace["load_model_kwargs"](str(run_dir / relative_path))
    assert kwargs == {
        "min_scale": 0.0005, "max_scale": 1.0, "scale_eps": 1e-5,
        "eps2d": 7e-8, "eps3d": 8e-9,
        "img_tile_size": 32, "vol_tile_size": 4,
        "img_radius_clip": 2.0, "vol_radius_clip": 2.5,
        "backward_mode": "pixel",
    }


def test_uses_nearest_run_config_instead_of_outer_config(lod_module, saved_config):
    run_dir, cfg = saved_config
    nested = run_dir / "student_run"
    nested.mkdir()
    cfg.model.scale_min = 0.002
    cfg.volume_epsilon = 2e-9
    OmegaConf.save(cfg, nested / "config_used.yaml")
    kwargs = lod_module.namespace["load_model_kwargs"](
        str(nested / "lod_1024/ckpts/point_cloud_29999.pickle")
    )
    assert kwargs["min_scale"] == 0.002
    assert kwargs["eps3d"] == 2e-9


@pytest.mark.parametrize("loader, relative_path", [
    ("load_teacher_model", "ckpts/ckpt_29999.pt"),
    ("load_lod_checkpoint_model", "lod/ckpt_lod_1024_step_100.pt"),
    ("load_lod_model", "lod_1024/ckpts/point_cloud_29999.pickle"),
])
def test_each_loader_passes_config_to_model(lod_module, saved_config, loader, relative_path):
    run_dir, _ = saved_config
    checkpoint = str(run_dir / relative_path)
    result = lod_module.namespace[loader](checkpoint, "cpu")
    factory = lod_module.binary_tree if loader == "load_teacher_model" else lod_module.gaussian
    factory.assert_called_once_with(
        device="cpu", min_scale=0.0005, max_scale=1.0, scale_eps=1e-5,
        eps2d=7e-8, eps3d=8e-9, img_tile_size=32, vol_tile_size=4,
        img_radius_clip=2.0, vol_radius_clip=2.5, backward_mode="pixel",
    )
    assert result is factory.return_value
    result.eval.assert_called_once_with()
    if loader == "load_teacher_model":
        result.load_ckpt.assert_called_once_with(checkpoint)
    elif loader == "load_lod_checkpoint_model":
        lod_module.torch_load.assert_called_once_with(checkpoint, map_location="cpu", weights_only=True)
        for name, tensor in lod_module.fields.items():
            assert getattr(result, name) is tensor
    else:
        result.load_ply.assert_called_once_with(checkpoint)
        result._configure_scale_bounds.assert_called_once_with(
            min_scale=0.0005, max_scale=1.0, scale_eps=1e-5,
        )


def test_missing_config_fails_before_loading_model(lod_module, tmp_path):
    with pytest.raises(FileNotFoundError, match="No Hydra config_used.yaml"):
        lod_module.namespace["load_lod_checkpoint_model"](str(tmp_path / "ckpts/missing.pt"), "cpu")
    lod_module.torch_load.assert_not_called()
    lod_module.gaussian.assert_not_called()


@pytest.mark.parametrize("missing_key", ["model", "eps3d", "scale_min"])
def test_incomplete_config_does_not_use_model_defaults(lod_module, saved_config, missing_key):
    run_dir, cfg = saved_config
    if missing_key == "model":
        del cfg.model
    else:
        del cfg.model[missing_key]
    OmegaConf.save(cfg, run_dir / "config_used.yaml")
    with pytest.raises(ValueError, match="model.*config|Missing model settings") as caught:
        lod_module.namespace["load_model_kwargs"](str(run_dir / "ckpts/model.pt"))
    assert str(run_dir / "config_used.yaml") in str(caught.value)


def test_unpaired_scale_bounds_are_rejected(lod_module, saved_config):
    run_dir, cfg = saved_config
    cfg.model.scale_min = None
    OmegaConf.save(cfg, run_dir / "config_used.yaml")
    with pytest.raises(ValueError, match="both be set or null"):
        lod_module.namespace["load_model_kwargs"](str(run_dir / "ckpts/model.pt"))


@pytest.mark.parametrize("saved_bounds", [None, [0.0005000000237487257, 1.0]])
def test_pickle_bounds_support_legacy_metadata_and_float_rounding(lod_module, saved_config, saved_bounds):
    run_dir, _ = saved_config
    model = lod_module.gaussian.return_value
    model._get_scale_bound.return_value = saved_bounds
    lod_module.namespace["load_lod_model"](str(run_dir / "lod_1024/ckpts/model.pickle"), "cpu")
    model._configure_scale_bounds.assert_called_once_with(
        min_scale=0.0005, max_scale=1.0, scale_eps=1e-5,
    )


def test_conflicting_pickle_bounds_are_rejected(lod_module, saved_config):
    run_dir, _ = saved_config
    model = lod_module.gaussian.return_value
    model._get_scale_bound.return_value = np.array([0.001, 1.0])
    with pytest.raises(ValueError, match="scale bounds disagree"):
        lod_module.namespace["load_lod_model"](str(run_dir / "lod_1024/ckpts/model.pickle"), "cpu")
    model._configure_scale_bounds.assert_not_called()
    model.eval.assert_not_called()


def test_eval_only_exports_all_model_metrics(lod_module, monkeypatch, tmp_path):
    import csv

    evaluate = lod_module.namespace["evaluate_models_only"]
    models = {name: SimpleNamespace(num_gaussians=n) for name, n in [
        ("init_lod_1024", 1100), ("final_lod_1024", 1100), ("teacher_leaf", 40000),
    ]}
    voxelize = Mock(side_effect=[object(), object(), object()])
    metric = Mock(side_effect=lambda model, *args, **kwargs: {
        "num_gaussians": model.num_gaussians,
        "psnr2d": 40.0, "ssim2d": 0.99, "psnr3d": 33.0, "ssim3d": 0.93,
    })
    monkeypatch.setitem(evaluate.__globals__, "voxelize_model", voxelize)
    monkeypatch.setitem(evaluate.__globals__, "evaluate_model", metric)
    metrics_dir = tmp_path / "render_output/metrics"
    results = evaluate(models, (2, 2, 2), None, object(), [object()], metrics_dir)
    assert list(results) == list(models)
    assert voxelize.call_count == metric.call_count == 3
    with (metrics_dir / "lod_metrics.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["model"] for row in rows] == list(models)
    assert all(row["psnr3d"] == "33.0000" for row in rows)
    assert list((tmp_path / "render_output").iterdir()) == [metrics_dir]


@pytest.mark.parametrize("failure", ["missing", "nan"])
def test_failed_reevaluation_preserves_previous_metrics(lod_module, monkeypatch, tmp_path, failure):
    evaluate = lod_module.namespace["evaluate_models_only"]
    monkeypatch.setitem(evaluate.__globals__, "voxelize_model", Mock(return_value=None))
    metrics = {"num_gaussians": 1000, "psnr2d": 40.0, "ssim2d": 0.99, "ssim3d": 0.9}
    if failure == "nan":
        metrics["psnr3d"] = float("nan")
    monkeypatch.setitem(evaluate.__globals__, "evaluate_model", Mock(return_value=metrics))
    previous = tmp_path / "lod_metrics.csv"
    previous.write_text("previous valid metrics\n")
    with pytest.raises(RuntimeError, match="Incomplete evaluation.*psnr3d"):
        evaluate({"teacher_leaf": object()}, (2, 2, 2), None, object(), [object()], tmp_path)
    assert previous.read_text() == "previous valid metrics\n"


def test_eval_only_main_returns_before_image_outputs(lod_module, monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    main = lod_module.namespace["main"]
    namespace = main.__globals__
    checkpoint = tmp_path / "lod_1024/ckpts/point_cloud_29999.pickle"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    out_dir = tmp_path / "render_output"
    ground_truth = SimpleNamespace(shape=(2, 2, 2))
    dataset = MagicMock()
    dataset.__len__.return_value = 1
    dataset.get_vol_gt.return_value = ground_truth
    model = SimpleNamespace(num_gaussians=1024)
    evaluate = Mock()
    overrides = {
        "torch": SimpleNamespace(device=lambda value: value, cuda=SimpleNamespace(is_available=lambda: False)),
        "CTImageParser": Mock(return_value=SimpleNamespace(bbox=None)),
        "CTImageDataset": Mock(return_value=dataset),
        "discover_lod_levels": Mock(return_value=[1024]),
        "discover_initial_lod_levels": Mock(return_value=[]),
        "load_lod_model": Mock(return_value=model),
        "evaluate_models_only": evaluate,
    }
    for key, value in overrides.items():
        monkeypatch.setitem(namespace, key, value)
    monkeypatch.setattr(sys, "argv", [
        "lod_render", "--lod_dir", str(tmp_path), "--data", "unused",
        "--out", str(out_dir), "--eval_only",
    ])
    main()
    evaluate.assert_called_once_with(
        {"final_lod_1024": model}, (2, 2, 2), None, ground_truth, dataset, out_dir / "metrics",
    )
    assert not out_dir.exists()


def test_eval_only_and_skip_eval_are_mutually_exclusive(lod_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "lod_render", "--lod_dir", "unused", "--data", "unused", "--out", "unused",
        "--eval_only", "--skip_eval",
    ])
    with pytest.raises(SystemExit) as caught:
        lod_module.namespace["main"]()
    assert caught.value.code == 2
