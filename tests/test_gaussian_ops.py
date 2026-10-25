import math
from types import SimpleNamespace

import pytest
import torch

from r2splat.ops import (
    InferenceGaussianState,
    MappingVoxelizationResult,
    ProjectionRasterizationResult,
)
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.models.gaussian_model import GaussianModel
from src.trainers.binary_tree import BinaryTreeTrainer
from src.utils.loss import compute_hierarchy_loss
from src.utils.math import invert_rigid_transform, normalized_quat_to_rotmat
from src.utils.ops import compute_relocation


def test_rigid_transform_inverse_matches_general_inverse():
    torch.manual_seed(11)
    quats = torch.nn.functional.normalize(
        torch.randn(5, 4, dtype=torch.float64), dim=-1
    )
    transforms = torch.eye(4, dtype=torch.float64).repeat(5, 1, 1)
    transforms[:, :3, :3] = normalized_quat_to_rotmat(quats)
    transforms[:, :3, 3] = torch.randn(5, 3, dtype=torch.float64)

    actual = invert_rigid_transform(transforms)
    expected = torch.linalg.inv(transforms)
    torch.testing.assert_close(actual, expected)


def test_inference_state_is_reusable_and_forward_only(monkeypatch):
    model = GaussianModel(device="cpu", min_scale=0.0005, max_scale=1.0)
    model.means = torch.nn.Parameter(torch.tensor([[0.0, 0.0, 4.0]]))
    model.scales = torch.nn.Parameter(torch.full((1, 3), -1.2))
    model.quats = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    model.opacities = torch.nn.Parameter(torch.tensor([-0.5]))
    prepared = InferenceGaussianState(
        covars=torch.ones(1, 6),
        max_scales=torch.ones(1),
        opacities=torch.ones(1),
    )
    seen_states = []

    def fake_prepare(quats, raw_scales, raw_opacities, *, scale_min, scale_max):
        assert quats is model.quats
        assert raw_scales is model.scales
        assert raw_opacities is model.opacities
        assert scale_min == pytest.approx(0.0005)
        assert scale_max == pytest.approx(1.0)
        return prepared

    def fake_rasterize(state, means, viewmats, _Ks, *, img_w, img_h, **_kwargs):
        seen_states.append(state)
        with torch.no_grad():
            value = means.sum() + state.covars.sum() + state.opacities.sum()
            image = value * torch.ones(
                (viewmats.shape[0], img_h, img_w), dtype=means.dtype
            )
            means2d = means[:, :2].unsqueeze(0).expand(viewmats.shape[0], -1, -1)
            radii = torch.ones(
                (viewmats.shape[0], means.shape[0], 2), dtype=means.dtype
            )
        return ProjectionRasterizationResult(
            image=image,
            means2d=means2d,
            radii=radii,
        )

    def fake_voxelize(
        state, means, _ranges, _centers, *, dim2, dim1, dim0, **_kwargs
    ):
        seen_states.append(state)
        with torch.no_grad():
            value = (
                state.covars.sum()
                + state.max_scales.sum()
                + state.opacities.sum()
            )
            volume = value * torch.ones((dim0, dim1, dim2), dtype=means.dtype)
            means3d = torch.zeros((means.shape[0], 3), dtype=means.dtype)
            radii = torch.ones((means.shape[0], 3), dtype=means.dtype)
        return MappingVoxelizationResult(
            volume=volume,
            means3d=means3d,
            radii=radii,
        )

    monkeypatch.setattr(
        "src.models.gaussian_model._prepare_inference_gaussians", fake_prepare
    )
    monkeypatch.setattr(
        "src.models.gaussian_model.inference_rasterize", fake_rasterize
    )
    monkeypatch.setattr("src.models.gaussian_model.inference_voxelize", fake_voxelize)

    state = model.prepare_gaussians()

    camtoworlds = torch.eye(4).unsqueeze(0)
    viewmats = invert_rigid_transform(camtoworlds)
    image, _ = model.rasterize(
        camtoworlds,
        torch.eye(3).unsqueeze(0),
        4,
        4,
        viewmats=viewmats,
        gaussian_state=state,
    )
    image_again, _ = model.rasterize(
        camtoworlds,
        torch.eye(3).unsqueeze(0),
        4,
        4,
        viewmats=viewmats,
        gaussian_state=state,
    )
    volume = model.voxelize(
        (4, 4, 4),
        torch.tensor([[-1.0, -1.0, 2.0], [1.0, 1.0, 6.0]]),
        gaussian_state=state,
    )

    assert state is prepared
    assert seen_states == [state, state, state]
    assert not image.requires_grad
    assert not image_again.requires_grad
    assert not volume.requires_grad


def test_relocation_matches_scalar_equation():
    opacities = torch.tensor([0.15, 0.4, 0.8], dtype=torch.float64)
    scales = torch.tensor(
        [[0.1, 0.2, 0.3], [0.5, 0.4, 0.3], [1.0, 0.8, 0.6]],
        dtype=torch.float64,
    )
    ratios = torch.tensor([1, 3, 7])
    n_max = 8
    binoms = torch.zeros(n_max, n_max, dtype=torch.float64)
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)

    actual_opacity, actual_scales = compute_relocation(
        opacities, scales, ratios, binoms
    )
    expected_opacity = torch.empty_like(opacities)
    expected_scales = torch.empty_like(scales)
    for idx, ratio in enumerate(ratios.tolist()):
        new_opacity = 1.0 - (1.0 - opacities[idx]) ** (1.0 / ratio)
        denominator = opacities.new_zeros(())
        for i in range(1, ratio + 1):
            for k in range(i):
                denominator += (
                    math.comb(i - 1, k)
                    * ((-1.0) ** k)
                    / math.sqrt(k + 1.0)
                    * new_opacity ** (k + 1)
                )
        expected_opacity[idx] = new_opacity
        expected_scales[idx] = scales[idx] * opacities[idx] / denominator

    torch.testing.assert_close(actual_opacity, expected_opacity)
    torch.testing.assert_close(actual_scales, expected_scales)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("packed", True, "packed=True"),
        ("sparse_grad", True, "sparse_grad=True"),
        ("rasterize_mode", "antialiased", "classic"),
        ("camera_model", "ortho", "pinhole"),
    ],
)
def test_model_rejects_unsupported_render_options(option, value, message):
    model = GaussianModel()
    kwargs = {option: value}
    with pytest.raises(ValueError, match=message):
        model.rasterize(
            torch.eye(4).unsqueeze(0),
            torch.eye(3).unsqueeze(0),
            width=16,
            height=16,
            **kwargs,
        )


def test_checkpoint_state_keys_are_backend_independent(tmp_path):
    model = GaussianModel(device="cpu")
    model.means = torch.nn.Parameter(torch.randn(4, 3))
    model.scales = torch.nn.Parameter(torch.randn(4, 3))
    model.quats = torch.nn.Parameter(torch.randn(4, 4))
    model.opacities = torch.nn.Parameter(torch.randn(4))
    path = tmp_path / "model.pt"
    model.save_ckpt(str(path), step=12)

    restored = GaussianModel(device="cpu")
    restored.means = torch.nn.Parameter(torch.empty(4, 3))
    restored.scales = torch.nn.Parameter(torch.empty(4, 3))
    restored.quats = torch.nn.Parameter(torch.empty(4, 4))
    restored.opacities = torch.nn.Parameter(torch.empty(4))
    assert restored.load_ckpt(str(path)) == 12
    assert restored.state_dict().keys() == model.state_dict().keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

    pickle_path = tmp_path / "point_cloud.pickle"
    model.save_ply(str(pickle_path))
    exported = GaussianModel(device="cpu")
    exported.load_ply(str(pickle_path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(exported.state_dict()[key], value)


def test_cached_hierarchy_loss_matches_reference_and_invalidates():
    torch.manual_seed(23)
    model = BinaryTreeGaussianModel(
        device="cpu", min_scale=0.0005, max_scale=1.0
    )
    model.means = torch.nn.Parameter(torch.randn(7, 3))
    model.parent = torch.tensor([-1, 0, 1, 2, 2, 3, 99], dtype=torch.int32)
    model.history_means = torch.randn(4, 3)
    model.history_scales = torch.randn(4, 3)
    model.history_quats = torch.randn(4, 4)

    trainer = BinaryTreeTrainer.__new__(BinaryTreeTrainer)
    trainer.model = model
    trainer.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            hierarchy_weight=0.05,
            hierarchy_parent_distance_factor=0.75,
        )
    )
    trainer._hierarchy_geometry_cache = None

    reference = compute_hierarchy_loss(
        model.means,
        model.parent,
        model.history_means,
        model.history_scales,
        model.history_quats,
        model.activate_scale,
        max_parent_distance_factor=0.75,
    )
    cached = trainer._compute_hierarchy_loss()
    reference_grad = torch.autograd.grad(reference, model.means, retain_graph=True)[0]
    cached_grad = torch.autograd.grad(cached, model.means)[0]
    torch.testing.assert_close(cached, reference)
    torch.testing.assert_close(cached_grad, reference_grad)

    previous_cache = trainer._hierarchy_geometry_cache
    model.history_means.add_(0.25)
    updated = trainer._compute_hierarchy_loss()
    updated_reference = compute_hierarchy_loss(
        model.means,
        model.parent,
        model.history_means,
        model.history_scales,
        model.history_quats,
        model.activate_scale,
        max_parent_distance_factor=0.75,
    )
    assert trainer._hierarchy_geometry_cache is not previous_cache
    torch.testing.assert_close(updated, updated_reference)
