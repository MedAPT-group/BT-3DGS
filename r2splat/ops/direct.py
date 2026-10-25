"""Direct CUDA forward operators for Gaussian CT splatting."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InferenceGaussianState:
    """Reusable transformed Gaussian parameters for forward-only rendering."""

    covars: torch.Tensor
    max_scales: torch.Tensor
    opacities: torch.Tensor


@dataclass
class ProjectionRasterizationResult:
    image: torch.Tensor
    means2d: torch.Tensor
    radii: torch.Tensor


@dataclass
class MappingVoxelizationResult:
    volume: torch.Tensor
    means3d: torch.Tensor
    radii: torch.Tensor


@dataclass
class ProjectionForwardBuffers:
    output: torch.Tensor
    means2d: torch.Tensor
    cov2d: torch.Tensor
    conics: torch.Tensor
    mus: torch.Tensor
    opacity: torch.Tensor
    radii: torch.Tensor
    tile_offsets: torch.Tensor
    flatten_ids: torch.Tensor
    isect_ids: torch.Tensor
    viewmats: torch.Tensor
    Ks: torch.Tensor
    img_w: int
    img_h: int
    tile_size: int
    eps2d: float


@dataclass
class VoxelForwardBuffers:
    output: torch.Tensor
    means3d: torch.Tensor
    conics: torch.Tensor
    opacity: torch.Tensor
    radii: torch.Tensor
    tile_offsets: torch.Tensor
    flatten_ids: torch.Tensor
    isect_ids: torch.Tensor
    ranges: torch.Tensor
    centers: torch.Tensor
    dim2: int
    dim1: int
    dim0: int
    tile_size: int
    eps3d: float


@torch.no_grad()
def _prepare_forward_parameters(
    quats: torch.Tensor,
    raw_scales: torch.Tensor,
    raw_opacities: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
) -> InferenceGaussianState:
    from ..cuda._wrapper import parameter_transform_fwd

    if not quats.is_cuda or not raw_scales.is_cuda or not raw_opacities.is_cuda:
        raise RuntimeError("Direct r2splat operators require CUDA tensors.")
    covars, max_scales = parameter_transform_fwd(
        quats.contiguous(),
        raw_scales.contiguous(),
        float(scale_min),
        float(scale_max),
    )
    opacities = torch.nn.functional.softplus(raw_opacities).contiguous()
    return InferenceGaussianState(
        covars=covars,
        max_scales=max_scales,
        opacities=opacities,
    )


def prepare_inference_gaussians(
    quats: torch.Tensor,
    raw_scales: torch.Tensor,
    raw_opacities: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
) -> InferenceGaussianState:
    """Transform raw parameters once for reusable forward-only evaluation."""
    return _prepare_forward_parameters(
        quats,
        raw_scales,
        raw_opacities,
        scale_min=scale_min,
        scale_max=scale_max,
    )


@torch.no_grad()
def _rasterize_forward(
    means: torch.Tensor,
    covars: torch.Tensor,
    opacities: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    *,
    img_w: int,
    img_h: int,
    tile_size: int,
    eps2d: float,
    radius_clip: float,
) -> ProjectionForwardBuffers:
    from ..cuda._wrapper import (
        conics_mu_fwd,
        isect_offset_encode,
        isect_tiles,
        rasterize_to_pixels_fwd,
        world_to_image_projection_fwd,
    )

    img_w = int(img_w)
    img_h = int(img_h)
    tile_size = int(tile_size)
    if tile_size <= 0 or tile_size * tile_size > 1024:
        raise ValueError("2D tile_size must be positive with tile_size**2 <= 1024.")

    means_c = means.contiguous()
    covars_c = covars.contiguous()
    viewmats_c = viewmats.contiguous()
    Ks_c = Ks.contiguous()
    means2d, cov2d, radii = world_to_image_projection_fwd(
        means_c,
        covars_c,
        viewmats_c,
        Ks_c,
        img_w,
        img_h,
        float(eps2d),
        float(radius_clip),
    )
    conics, mus = conics_mu_fwd(cov2d, float(eps2d))
    n_cameras = int(viewmats_c.shape[0])
    opacity = opacities.unsqueeze(0).expand(n_cameras, -1).contiguous()
    n_tiles_h = (img_h + tile_size - 1) // tile_size
    n_tiles_w = (img_w + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, tile_size, n_tiles_w, n_tiles_h
    )
    tile_offsets = isect_offset_encode(
        isect_ids, n_cameras, n_tiles_w, n_tiles_h
    )
    output = rasterize_to_pixels_fwd(
        means2d,
        conics,
        opacity,
        mus,
        tile_offsets,
        flatten_ids,
        img_w,
        img_h,
        tile_size,
    )
    return ProjectionForwardBuffers(
        output=output,
        means2d=means2d,
        cov2d=cov2d,
        conics=conics,
        mus=mus,
        opacity=opacity,
        radii=radii,
        tile_offsets=tile_offsets,
        flatten_ids=flatten_ids,
        isect_ids=isect_ids,
        viewmats=viewmats_c,
        Ks=Ks_c,
        img_w=img_w,
        img_h=img_h,
        tile_size=tile_size,
        eps2d=float(eps2d),
    )


def inference_rasterize(
    state: InferenceGaussianState,
    means: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    *,
    img_w: int,
    img_h: int,
    tile_size: int,
    eps2d: float,
    radius_clip: float,
) -> ProjectionRasterizationResult:
    """Rasterize without building or retaining any backward state."""
    buffers = _rasterize_forward(
        means,
        state.covars,
        state.opacities,
        viewmats,
        Ks,
        img_w=img_w,
        img_h=img_h,
        tile_size=tile_size,
        eps2d=eps2d,
        radius_clip=radius_clip,
    )
    return ProjectionRasterizationResult(
        image=buffers.output,
        means2d=buffers.means2d,
        radii=buffers.radii,
    )


@torch.no_grad()
def _voxelize_forward(
    means: torch.Tensor,
    covars: torch.Tensor,
    max_scales: torch.Tensor,
    opacities: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    *,
    dim2: int,
    dim1: int,
    dim0: int,
    tile_size: int,
    eps3d: float,
    radius_clip: float,
) -> VoxelForwardBuffers:
    from ..cuda._wrapper import (
        isect_offset_encode_3d,
        isect_tiles_3d,
        voxelize_to_voxels_fwd,
        world_to_volume_mapping_fwd,
    )

    dim2 = int(dim2)
    dim1 = int(dim1)
    dim0 = int(dim0)
    tile_size = int(tile_size)
    if tile_size <= 0 or tile_size**3 > 1024:
        raise ValueError("3D tile_size must be positive with tile_size**3 <= 1024.")

    means_c = means.contiguous()
    covars_c = covars.contiguous()
    max_scales_c = max_scales.contiguous()
    ranges_c = ranges.contiguous()
    centers_c = centers.contiguous()
    means3d, conics, radii = world_to_volume_mapping_fwd(
        means_c,
        covars_c,
        max_scales_c,
        ranges_c,
        centers_c,
        dim2,
        dim1,
        dim0,
        float(eps3d),
        float(radius_clip),
    )
    n_tiles_dim2 = (dim2 + tile_size - 1) // tile_size
    n_tiles_dim1 = (dim1 + tile_size - 1) // tile_size
    n_tiles_dim0 = (dim0 + tile_size - 1) // tile_size
    _, isect_ids, flatten_ids = isect_tiles_3d(
        means3d,
        radii,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0,
    )
    tile_offsets = isect_offset_encode_3d(
        isect_ids, n_tiles_dim2, n_tiles_dim1, n_tiles_dim0
    )
    opacity = opacities.contiguous()
    output = voxelize_to_voxels_fwd(
        means3d,
        conics,
        opacity,
        tile_offsets,
        flatten_ids,
        dim2,
        dim1,
        dim0,
        tile_size,
    )
    return VoxelForwardBuffers(
        output=output,
        means3d=means3d,
        conics=conics,
        opacity=opacity,
        radii=radii,
        tile_offsets=tile_offsets,
        flatten_ids=flatten_ids,
        isect_ids=isect_ids,
        ranges=ranges_c,
        centers=centers_c,
        dim2=dim2,
        dim1=dim1,
        dim0=dim0,
        tile_size=tile_size,
        eps3d=float(eps3d),
    )


def inference_voxelize(
    state: InferenceGaussianState,
    means: torch.Tensor,
    ranges: torch.Tensor,
    centers: torch.Tensor,
    *,
    dim2: int,
    dim1: int,
    dim0: int,
    tile_size: int,
    eps3d: float,
    radius_clip: float,
) -> MappingVoxelizationResult:
    """Voxelize without building or retaining any backward state."""
    buffers = _voxelize_forward(
        means,
        state.covars,
        state.max_scales,
        state.opacities,
        ranges,
        centers,
        dim2=dim2,
        dim1=dim1,
        dim0=dim0,
        tile_size=tile_size,
        eps3d=eps3d,
        radius_clip=radius_clip,
    )
    return MappingVoxelizationResult(
        volume=buffers.output,
        means3d=buffers.means3d,
        radii=buffers.radii,
    )


__all__ = [
    "InferenceGaussianState",
    "MappingVoxelizationResult",
    "ProjectionRasterizationResult",
    "inference_rasterize",
    "inference_voxelize",
    "prepare_inference_gaussians",
]
