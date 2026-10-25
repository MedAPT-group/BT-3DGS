import pytest
import torch

from r2splat.cuda import is_cuda_extension_available
from r2splat.cuda._wrapper import (
    conics_mu_bwd,
    conics_mu_fwd,
    parameter_transform_bwd,
    parameter_transform_fwd,
    rasterize_to_pixels_bwd,
    rasterize_to_pixels_bwd_per_gaussian,
    rasterize_to_pixels_fwd,
    voxelize_to_voxels_bwd,
    voxelize_to_voxels_bwd_per_gaussian,
    voxelize_to_voxels_fwd,
)
from src.models.binary_tree_gaussian_model import BinaryTreeGaussianModel
from src.models.gaussian_model import GaussianModel
from src.models.student_gaussian_model import StudentGaussianModel


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


def test_cuda_extension_is_installed():
    assert is_cuda_extension_available(), "r2splat CUDA extension failed to import"


def _covars_from_quats_scales_reference(quats, scales):
    rotations = torch.nn.functional.normalize(quats, dim=-1)
    w, x, y, z = rotations.unbind(dim=-1)
    matrix = torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    full = matrix @ torch.diag_embed(scales.square()) @ matrix.transpose(-1, -2)
    return full[:, (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def test_parameter_transform_matches_torch_for_raw_quaternions():
    torch.manual_seed(19)
    quats = torch.randn(11, 4, device="cuda") * 0.7
    raw_scales = torch.randn(11, 3, device="cuda")
    scale_min, scale_max = 0.0005, 1.0

    covars, max_scales = parameter_transform_fwd(
        quats.contiguous(), raw_scales.contiguous(), scale_min, scale_max
    )
    scales_ref = scale_min + (scale_max - scale_min) * torch.sigmoid(raw_scales)
    covars_ref = _covars_from_quats_scales_reference(quats, scales_ref)
    torch.testing.assert_close(covars, covars_ref, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(
        max_scales, scales_ref.amax(dim=-1), atol=3e-4, rtol=3e-4
    )

    v_covars = torch.randn_like(covars)
    v_quats, v_raw_scales = parameter_transform_bwd(
        quats.contiguous(),
        raw_scales.contiguous(),
        scale_min,
        scale_max,
        v_covars.contiguous(),
    )
    q_ref = quats.detach().clone().requires_grad_(True)
    s_ref = raw_scales.detach().clone().requires_grad_(True)
    scales_ref = scale_min + (scale_max - scale_min) * torch.sigmoid(s_ref)
    covars_ref = _covars_from_quats_scales_reference(q_ref, scales_ref)
    expected = torch.autograd.grad((covars_ref * v_covars).sum(), (q_ref, s_ref))
    torch.testing.assert_close(v_quats, expected[0], atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(v_raw_scales, expected[1], atol=3e-4, rtol=3e-4)


def _unpack_covariance_reference(covars, eps):
    indices = torch.tensor(
        [[0, 1, 2], [1, 3, 4], [2, 4, 5]], device=covars.device
    )
    return covars[..., indices] + eps * torch.eye(
        3, dtype=covars.dtype, device=covars.device
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("eps", [0.0, 1e-8])
@pytest.mark.parametrize("mu_only", [True, False])
def test_conics_mu_preserves_gradients_for_small_positive_determinants(
    dtype, eps, mu_only
):
    # Both diagonal and correlated covariances span the old D3=1e-9 cutoff.
    covars = torch.tensor(
        [
            [
                [1e-2, 0.0, 0.0, 1e-2, 0.0, f]
                for f in (1e-6, 9e-6, 11e-6, 1e-4)
            ],
            [
                [1e-2, 2e-3, 1e-5, 2e-2, -2e-5, f]
                for f in (1e-6, 4e-6, 6e-6, 1e-4)
            ],
        ],
        dtype=dtype,
        device="cuda",
    )
    conics, mus = conics_mu_fwd(covars, eps)
    reference = covars.double().detach().requires_grad_(True)
    covariance = _unpack_covariance_reference(reference, eps)
    covariance2d = covariance[..., :2, :2]
    determinant2d = torch.linalg.det(covariance2d)
    determinant3d = torch.linalg.det(covariance)
    assert (determinant2d > 1e-9).all()
    assert (determinant3d > 0).all()
    assert (determinant3d < 1e-9).any() and (determinant3d > 1e-9).any()
    inverse2d = torch.linalg.inv(covariance2d)
    conics_ref = inverse2d[..., (0, 0, 1), (0, 1, 1)]
    mus_ref = (2.0 * torch.pi * determinant3d / determinant2d).sqrt()

    v_conics = torch.zeros_like(conics)
    if not mu_only:
        v_conics = torch.linspace(
            -0.2, 0.3, conics.numel(), dtype=dtype, device="cuda"
        ).reshape_as(conics)
    v_mus = torch.linspace(
        0.5, 1.2, mus.numel(), dtype=dtype, device="cuda"
    ).reshape_as(mus)
    actual_grad = conics_mu_bwd(covars, eps, conics, mus, v_conics, v_mus)
    expected_grad = torch.autograd.grad(
        (conics_ref * v_conics.double()).sum() + (mus_ref * v_mus.double()).sum(),
        reference,
    )[0]

    tolerance = 3e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(
        conics, conics_ref.to(dtype), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(mus, mus_ref.to(dtype), atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        actual_grad, expected_grad.to(dtype), atol=tolerance, rtol=tolerance
    )
    assert (mus > 0).all()
    assert (actual_grad[..., 5] > 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("last_diagonal", [0.0, -1e-6])
def test_conics_mu_nonpositive_determinant_keeps_only_conic_gradients(
    dtype, last_diagonal
):
    covars = torch.tensor(
        [[[1e-2, 2e-3, 0.0, 2e-2, 0.0, last_diagonal]]],
        dtype=dtype,
        device="cuda",
    )
    conics, mus = conics_mu_fwd(covars, 0.0)
    v_conics = torch.ones_like(conics)
    actual_grad = conics_mu_bwd(
        covars, 0.0, conics, mus, v_conics, torch.ones_like(mus)
    )
    reference = covars.double().detach().requires_grad_(True)
    covariance = _unpack_covariance_reference(reference, 0.0)
    inverse2d = torch.linalg.inv(covariance[..., :2, :2])
    conics_ref = inverse2d[..., (0, 0, 1), (0, 1, 1)]
    expected_grad = torch.autograd.grad(conics_ref.sum(), reference)[0]

    tolerance = 3e-4 if dtype == torch.float32 else 1e-9
    torch.testing.assert_close(
        conics, conics_ref.to(dtype), atol=tolerance, rtol=tolerance
    )
    torch.testing.assert_close(mus, torch.zeros_like(mus))
    torch.testing.assert_close(
        actual_grad, expected_grad.to(dtype), atol=tolerance, rtol=tolerance
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conics_mu_rejects_degenerate_2d_covariance(dtype):
    covars = torch.tensor(
        [[[1e-6, 0.0, 0.0, 1e-6, 0.0, 1.0]]], dtype=dtype, device="cuda"
    )
    conics, mus = conics_mu_fwd(covars, 1e-8)
    actual_grad = conics_mu_bwd(
        covars, 1e-8, conics, mus, torch.ones_like(conics), torch.ones_like(mus)
    )
    torch.testing.assert_close(conics, torch.zeros_like(conics))
    torch.testing.assert_close(mus, torch.zeros_like(mus))
    torch.testing.assert_close(actual_grad, torch.zeros_like(covars))


def _raster_reference(means2d, conics, opacity, mus, height, width):
    py, px = torch.meshgrid(
        torch.arange(height, device=means2d.device, dtype=means2d.dtype) + 0.5,
        torch.arange(width, device=means2d.device, dtype=means2d.dtype) + 0.5,
        indexing="ij",
    )
    dx = px[None, None] - means2d[..., 0, None, None]
    dy = py[None, None] - means2d[..., 1, None, None]
    sigma = (
        0.5 * conics[..., 0, None, None] * dx.square()
        + conics[..., 1, None, None] * dx * dy
        + 0.5 * conics[..., 2, None, None] * dy.square()
    )
    alpha = opacity[..., None, None] * mus[..., None, None] * torch.exp(-sigma)
    alpha = torch.where((sigma >= 0) & (alpha.abs() >= 1.0e-5), alpha, 0.0)
    return alpha.sum(dim=1)


def _voxel_reference(means3d, conics, opacity, dim0, dim1, dim2):
    pz, py, px = torch.meshgrid(
        torch.arange(dim0, device=means3d.device, dtype=means3d.dtype) + 0.5,
        torch.arange(dim1, device=means3d.device, dtype=means3d.dtype) + 0.5,
        torch.arange(dim2, device=means3d.device, dtype=means3d.dtype) + 0.5,
        indexing="ij",
    )
    dx = px[None] - means3d[:, 0, None, None, None]
    dy = py[None] - means3d[:, 1, None, None, None]
    dz = pz[None] - means3d[:, 2, None, None, None]
    sigma = (
        0.5 * conics[:, 0, None, None, None] * dx.square()
        + conics[:, 1, None, None, None] * dx * dy
        + conics[:, 2, None, None, None] * dx * dz
        + 0.5 * conics[:, 3, None, None, None] * dy.square()
        + conics[:, 4, None, None, None] * dy * dz
        + 0.5 * conics[:, 5, None, None, None] * dz.square()
    )
    alpha = opacity[:, None, None, None] * torch.exp(-sigma)
    alpha = torch.where((sigma >= 0) & (alpha.abs() >= 1.0e-6), alpha, 0.0)
    return alpha.sum(dim=0)


@pytest.mark.parametrize("primitive_backward", [False, True])
def test_raster_forward_backward_matches_torch(primitive_backward):
    device = "cuda"
    means = torch.tensor(
        [[[5.2, 6.1], [10.0, 9.3]]], device=device, requires_grad=True
    )
    conics = torch.tensor(
        [[[0.18, 0.01, 0.15], [0.12, -0.015, 0.2]]],
        device=device,
        requires_grad=True,
    )
    opacity = torch.tensor([[0.7, 0.4]], device=device, requires_grad=True)
    mus = torch.tensor([[1.2, 0.8]], device=device, requires_grad=True)
    offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.arange(2, dtype=torch.int32, device=device)
    isect_ids = torch.zeros(2, dtype=torch.int64, device=device)
    actual = rasterize_to_pixels_fwd(
        means,
        conics,
        opacity,
        mus,
        offsets,
        flatten_ids,
        16,
        16,
        16,
    )
    expected = _raster_reference(means, conics, opacity, mus, 16, 16)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    weight = torch.linspace(0.1, 1.0, actual.numel(), device=device).reshape_as(
        actual
    )
    if primitive_backward:
        direct_grads = rasterize_to_pixels_bwd_per_gaussian(
            means,
            conics,
            opacity,
            mus,
            offsets,
            isect_ids,
            flatten_ids,
            weight.contiguous(),
            16,
            16,
            16,
            True,
        )
    else:
        direct_grads = rasterize_to_pixels_bwd(
            means,
            conics,
            opacity,
            mus,
            offsets,
            flatten_ids,
            weight.contiguous(),
            16,
            16,
            16,
            True,
        )
    expected_grads = torch.autograd.grad(
        (expected * weight).sum(), (means, conics, opacity, mus)
    )
    for actual_grad, expected_grad in zip(direct_grads[1:], expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-4, rtol=3e-4)
    assert direct_grads[0].shape == means.shape
    assert torch.isfinite(direct_grads[0]).all()


@pytest.mark.parametrize("primitive_backward", [False, True])
def test_voxel_forward_backward_matches_torch(primitive_backward):
    device = "cuda"
    means = torch.tensor(
        [[2.5, 3.1, 4.0], [5.2, 4.6, 3.3]],
        device=device,
        requires_grad=True,
    )
    conics = torch.tensor(
        [
            [0.3, 0.01, 0.0, 0.25, -0.01, 0.2],
            [0.22, 0.0, 0.01, 0.28, 0.0, 0.24],
        ],
        device=device,
        requires_grad=True,
    )
    opacity = torch.tensor([0.8, 0.5], device=device, requires_grad=True)
    offsets = torch.zeros((1, 1, 1), dtype=torch.int32, device=device)
    flatten_ids = torch.arange(2, dtype=torch.int32, device=device)
    isect_ids = torch.zeros(2, dtype=torch.int64, device=device)
    actual = voxelize_to_voxels_fwd(
        means,
        conics,
        opacity,
        offsets,
        flatten_ids,
        8,
        8,
        8,
        8,
    )
    expected = _voxel_reference(means, conics, opacity, 8, 8, 8)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)

    weight = torch.linspace(0.1, 1.0, actual.numel(), device=device).reshape_as(
        actual
    )
    if primitive_backward:
        actual_grads = voxelize_to_voxels_bwd_per_gaussian(
            means,
            conics,
            opacity,
            offsets,
            isect_ids,
            flatten_ids,
            weight.contiguous(),
            8,
            8,
            8,
            8,
        )
    else:
        actual_grads = voxelize_to_voxels_bwd(
            means,
            conics,
            opacity,
            offsets,
            flatten_ids,
            weight.contiguous(),
            8,
            8,
            8,
            8,
        )
    expected_grads = torch.autograd.grad(
        (expected * weight).sum(), (means, conics, opacity)
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-4, rtol=3e-4)


def test_model_inference_projection_and_voxelization_are_forward_only():
    model = GaussianModel(
        device="cuda",
        min_scale=0.0005,
        max_scale=1.0,
        backward_mode="primitive",
    )
    model.means = torch.nn.Parameter(
        torch.tensor([[0.0, 0.0, 4.0]], device="cuda")
    )
    model.scales = torch.nn.Parameter(torch.full((1, 3), -1.2, device="cuda"))
    model.quats = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    )
    model.opacities = torch.nn.Parameter(torch.tensor([-0.5], device="cuda"))
    camtoworlds = torch.eye(4, device="cuda").unsqueeze(0)
    Ks = torch.tensor(
        [[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )

    state = model.prepare_gaussians()
    image, info = model.rasterize(
        camtoworlds, Ks, 16, 16, gaussian_state=state
    )
    image_again, _ = model.rasterize(
        camtoworlds, Ks, 16, 16, gaussian_state=state
    )
    assert image.shape == (1, 16, 16, 1)
    assert image.dtype == model.means.dtype and image.device == model.means.device
    assert torch.isfinite(image).all()
    torch.testing.assert_close(image_again, image)
    assert not image.requires_grad
    assert info["radii"].shape == (1, 1)
    assert info["gaussian_ids"].shape == (1, 1)

    volume = model.voxelize(
        (8, 8, 8),
        torch.tensor([[-2.0, -2.0, 2.0], [2.0, 2.0, 6.0]], device="cuda"),
        gaussian_state=state,
    )
    assert volume.shape == (8, 8, 8, 1)
    assert volume.dtype == model.means.dtype and volume.device == model.means.device
    assert torch.isfinite(volume).all()
    assert not volume.requires_grad


def _make_gradient_test_model() -> GaussianModel:
    model = GaussianModel(
        device="cuda",
        min_scale=0.0005,
        max_scale=1.0,
        eps2d=1.0e-8,
        eps3d=1.0e-8,
        backward_mode="primitive",
    )
    model.means = torch.nn.Parameter(
        torch.tensor([[0.0, 0.0, 4.0], [0.35, -0.2, 4.4]], device="cuda")
    )
    model.scales = torch.nn.Parameter(
        torch.tensor([[-1.2, -1.0, -0.8], [-0.9, -1.3, -1.1]], device="cuda")
    )
    model.quats = torch.nn.Parameter(
        torch.tensor(
            [[0.51, 0.08, -0.03, 0.02], [0.72, -0.12, 0.06, 0.09]],
            device="cuda",
        )
    )
    model.opacities = torch.nn.Parameter(torch.tensor([-0.5, -0.2], device="cuda"))
    return model


def test_explicit_training_matches_inference_forward():
    reference = _make_gradient_test_model()
    explicit = _make_gradient_test_model()
    explicit.load_state_dict(reference.state_dict())
    camtoworlds = torch.eye(4, device="cuda").unsqueeze(0)
    Ks = torch.tensor(
        [[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    bbox = torch.tensor(
        [[-2.0, -2.0, 2.0], [2.0, 2.0, 6.0]], device="cuda"
    )

    reference_state = reference.prepare_gaussians()
    reference_image, _ = reference.rasterize(
        camtoworlds,
        Ks,
        16,
        16,
        gaussian_state=reference_state,
    )
    reference_volume = reference.voxelize(
        (8, 8, 8), bbox, gaussian_state=reference_state
    )
    assert not reference_image.requires_grad
    assert not reference_volume.requires_grad

    explicit_state = explicit.prepare_gaussians(explicit=True)
    explicit_image, explicit_info = explicit.rasterize(
        camtoworlds,
        Ks,
        16,
        16,
        absgrad=True,
        gaussian_state=explicit_state,
    )
    explicit_volume = explicit.voxelize(
        (8, 8, 8), bbox, gaussian_state=explicit_state
    )
    explicit_info["means2d"].retain_grad()
    (
        explicit_image.square().mean()
        + 0.03 * explicit_volume.square().mean()
        + 0.01 * explicit.means.square().mean()
    ).backward()
    explicit.backward_explicit(explicit_state)

    torch.testing.assert_close(explicit_image, reference_image, atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(explicit_volume, reference_volume, atol=3e-4, rtol=3e-4)
    for name in ("means", "scales", "quats", "opacities"):
        grad = getattr(explicit, name).grad
        assert grad is not None
        assert torch.isfinite(grad).all()
    assert explicit_info["means2d"].grad is not None
    assert torch.isfinite(explicit_info["means2d"].grad).all()
    assert explicit_info["means2d"].absgrad is not None
    assert torch.isfinite(explicit_info["means2d"].absgrad).all()


@pytest.mark.parametrize(
    "model_type",
    [GaussianModel, BinaryTreeGaussianModel, StudentGaussianModel],
    ids=["default", "binary-tree", "distill-and-lod"],
)
def test_training_model_variants_complete_one_projection_step(model_type):
    model = model_type(
        device="cuda",
        min_scale=0.0005,
        max_scale=1.0,
        backward_mode="primitive",
    )
    model.means = torch.nn.Parameter(
        torch.tensor([[0.0, 0.0, 4.0]], device="cuda")
    )
    model.scales = torch.nn.Parameter(torch.full((1, 3), -1.2, device="cuda"))
    model.quats = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda")
    )
    model.opacities = torch.nn.Parameter(torch.tensor([-0.5], device="cuda"))
    if isinstance(model, BinaryTreeGaussianModel):
        model.trainable_param_keys = ["means", "scales", "quats", "opacities"]
    optimizers = model.build_optimizers(
        {"means": 1e-4, "scales": 1e-4, "quats": 1e-4, "opacities": 1e-4}
    )
    camtoworlds = torch.eye(4, device="cuda").unsqueeze(0)
    Ks = torch.tensor(
        [[[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]]],
        device="cuda",
    )
    gaussian_state = model.prepare_gaussians(explicit=True)
    image, info = model.rasterize(
        camtoworlds,
        Ks,
        16,
        16,
        absgrad=True,
        gaussian_state=gaussian_state,
    )
    info["means2d"].retain_grad()
    image.square().mean().backward()
    model.backward_explicit(gaussian_state)
    for optimizer in optimizers.values():
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    assert info["means2d"].grad is not None
    assert info["means2d"].absgrad is not None
