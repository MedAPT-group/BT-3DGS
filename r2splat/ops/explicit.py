"""Explicit first-order Gaussian CT splatting.

The CUDA forward pass is deliberately detached from PyTorch autograd.  The
rendered image and volume are exposed as leaf tensors so arbitrary PyTorch
losses can compute their output gradients.  :func:`explicit_backward` then
applies the Gaussian CUDA VJP once and accumulates gradients into the raw model
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .direct import (
    MappingVoxelizationResult,
    ProjectionRasterizationResult,
    _prepare_forward_parameters,
    _rasterize_forward,
    _voxelize_forward,
)


def _validate_backward_mode(backward_mode: str) -> str:
    mode = str(backward_mode).lower()
    if mode not in {"pixel", "primitive"}:
        raise ValueError("backward_mode must be 'pixel' or 'primitive'.")
    return mode


@dataclass
class ExplicitProjectionContext:
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
    absgrad: bool


@dataclass
class ExplicitVoxelContext:
    output: torch.Tensor
    means3d: torch.Tensor
    conics: torch.Tensor
    opacity: torch.Tensor
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


@dataclass
class ExplicitGaussianState:
    """One-shot context shared by projection and voxelization for one step."""

    covars: torch.Tensor
    max_scales: torch.Tensor
    opacities: torch.Tensor
    scale_min: float
    scale_max: float
    backward_mode: str
    projection: Optional[ExplicitProjectionContext] = None
    voxel: Optional[ExplicitVoxelContext] = None
    backward_called: bool = False


def prepare_explicit_gaussians(
    quats: torch.Tensor,
    raw_scales: torch.Tensor,
    raw_opacities: torch.Tensor,
    *,
    scale_min: float,
    scale_max: float,
    backward_mode: str = "primitive",
) -> ExplicitGaussianState:
    """Activate raw parameters without constructing a renderer autograd graph."""
    mode = _validate_backward_mode(backward_mode)
    prepared = _prepare_forward_parameters(
        quats,
        raw_scales,
        raw_opacities,
        scale_min=scale_min,
        scale_max=scale_max,
    )
    return ExplicitGaussianState(
        covars=prepared.covars,
        max_scales=prepared.max_scales,
        opacities=prepared.opacities,
        scale_min=float(scale_min),
        scale_max=float(scale_max),
        backward_mode=mode,
    )


def explicit_rasterize(
    state: ExplicitGaussianState,
    means: torch.Tensor,
    viewmats: torch.Tensor,
    Ks: torch.Tensor,
    *,
    img_w: int,
    img_h: int,
    tile_size: int,
    eps2d: float,
    radius_clip: float,
    absgrad: bool,
) -> ProjectionRasterizationResult:
    """Run the Gaussian projection/rasterization forward path explicitly."""
    if state.projection is not None:
        raise RuntimeError("An explicit Gaussian state can only be rasterized once.")
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
    output = buffers.output.detach().requires_grad_(True)
    means2d = buffers.means2d.detach().requires_grad_(True)
    state.projection = ExplicitProjectionContext(
        output=output,
        means2d=means2d,
        cov2d=buffers.cov2d,
        conics=buffers.conics,
        mus=buffers.mus,
        opacity=buffers.opacity,
        radii=buffers.radii,
        tile_offsets=buffers.tile_offsets,
        flatten_ids=buffers.flatten_ids,
        isect_ids=buffers.isect_ids,
        viewmats=buffers.viewmats,
        Ks=buffers.Ks,
        img_w=buffers.img_w,
        img_h=buffers.img_h,
        tile_size=buffers.tile_size,
        eps2d=buffers.eps2d,
        absgrad=bool(absgrad),
    )
    return ProjectionRasterizationResult(
        image=output,
        means2d=means2d,
        radii=buffers.radii,
    )


def explicit_voxelize(
    state: ExplicitGaussianState,
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
    """Run the Gaussian mapping/voxelization forward path explicitly."""
    if state.voxel is not None:
        raise RuntimeError("An explicit Gaussian state can only be voxelized once.")
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
    output = buffers.output.detach().requires_grad_(True)
    state.voxel = ExplicitVoxelContext(
        output=output,
        means3d=buffers.means3d,
        conics=buffers.conics,
        opacity=buffers.opacity,
        tile_offsets=buffers.tile_offsets,
        flatten_ids=buffers.flatten_ids,
        isect_ids=buffers.isect_ids,
        ranges=buffers.ranges,
        centers=buffers.centers,
        dim2=buffers.dim2,
        dim1=buffers.dim1,
        dim0=buffers.dim0,
        tile_size=buffers.tile_size,
        eps3d=buffers.eps3d,
    )
    return MappingVoxelizationResult(
        volume=output,
        means3d=buffers.means3d,
        radii=buffers.radii,
    )


def _accumulate_grad(parameter: torch.Tensor, grad: torch.Tensor) -> None:
    grad = grad.detach()
    if parameter.grad is None:
        parameter.grad = grad
    else:
        parameter.grad.add_(grad)


@torch.no_grad()
def explicit_backward(
    state: ExplicitGaussianState,
    means: torch.Tensor,
    quats: torch.Tensor,
    raw_scales: torch.Tensor,
    raw_opacities: torch.Tensor,
) -> None:
    """Apply the saved first-order CUDA VJP and accumulate raw parameter grads."""
    from ..cuda._wrapper import (
        conics_mu_bwd,
        parameter_transform_bwd,
        rasterize_to_pixels_bwd,
        rasterize_to_pixels_bwd_per_gaussian,
        voxelize_to_voxels_bwd,
        voxelize_to_voxels_bwd_per_gaussian,
        world_to_image_projection_bwd,
        world_to_volume_mapping_bwd,
    )

    if state.backward_called:
        raise RuntimeError("Explicit first-order backward can only run once per state.")
    if state.projection is None:
        raise RuntimeError("Explicit backward requires a projection forward pass.")

    v_means_total = torch.zeros_like(means)
    v_covars_total = torch.zeros_like(state.covars)
    v_opacity_total = torch.zeros_like(state.opacities)

    projection = state.projection
    v_image = projection.output.grad
    if v_image is None:
        v_means2d = torch.zeros_like(projection.means2d)
        v_means2d_abs = (
            torch.zeros_like(projection.means2d) if projection.absgrad else None
        )
    else:
        raster_args = (
            projection.means2d,
            projection.conics,
            projection.opacity,
            projection.mus,
            projection.tile_offsets,
        )
        if state.backward_mode == "primitive":
            raster_grads = rasterize_to_pixels_bwd_per_gaussian(
                *raster_args,
                projection.isect_ids,
                projection.flatten_ids,
                v_image.contiguous(),
                projection.img_w,
                projection.img_h,
                projection.tile_size,
                projection.absgrad,
            )
        else:
            raster_grads = rasterize_to_pixels_bwd(
                *raster_args,
                projection.flatten_ids,
                v_image.contiguous(),
                projection.img_w,
                projection.img_h,
                projection.tile_size,
                projection.absgrad,
            )
        v_means2d_abs, v_means2d, v_conics, v_opacity, v_mus = raster_grads
        v_cov2d = conics_mu_bwd(
            projection.cov2d,
            projection.eps2d,
            projection.conics,
            projection.mus,
            v_conics.contiguous(),
            v_mus.contiguous(),
        )
        v_means, v_covars = world_to_image_projection_bwd(
            means.contiguous(),
            state.covars,
            projection.viewmats,
            projection.Ks,
            projection.img_w,
            projection.img_h,
            v_means2d.contiguous(),
            v_cov2d.contiguous(),
        )
        v_means_total.add_(v_means)
        v_covars_total.add_(v_covars)
        v_opacity_total.add_(v_opacity.sum(dim=0))

    projection.means2d.grad = v_means2d.detach()
    if projection.absgrad:
        projection.means2d.absgrad = v_means2d_abs.detach()

    voxel = state.voxel
    if voxel is not None and voxel.output.grad is not None:
        voxel_args = (
            voxel.means3d,
            voxel.conics,
            voxel.opacity,
            voxel.tile_offsets,
        )
        if state.backward_mode == "primitive":
            v_means3d, v_conics3d, v_opacity3d = (
                voxelize_to_voxels_bwd_per_gaussian(
                    *voxel_args,
                    voxel.isect_ids,
                    voxel.flatten_ids,
                    voxel.output.grad.contiguous(),
                    voxel.dim2,
                    voxel.dim1,
                    voxel.dim0,
                    voxel.tile_size,
                )
            )
        else:
            v_means3d, v_conics3d, v_opacity3d = voxelize_to_voxels_bwd(
                *voxel_args,
                voxel.flatten_ids,
                voxel.output.grad.contiguous(),
                voxel.dim2,
                voxel.dim1,
                voxel.dim0,
                voxel.tile_size,
            )
        v_means, v_covars = world_to_volume_mapping_bwd(
            means.contiguous(),
            state.covars,
            voxel.ranges,
            voxel.centers,
            voxel.dim2,
            voxel.dim1,
            voxel.dim0,
            voxel.eps3d,
            v_means3d.contiguous(),
            v_conics3d.contiguous(),
        )
        v_means_total.add_(v_means)
        v_covars_total.add_(v_covars)
        v_opacity_total.add_(v_opacity3d)

    v_quats, v_raw_scales = parameter_transform_bwd(
        quats.contiguous(),
        raw_scales.contiguous(),
        state.scale_min,
        state.scale_max,
        v_covars_total.contiguous(),
    )
    v_raw_opacities = v_opacity_total * torch.sigmoid(raw_opacities)

    _accumulate_grad(means, v_means_total)
    _accumulate_grad(quats, v_quats)
    _accumulate_grad(raw_scales, v_raw_scales)
    _accumulate_grad(raw_opacities, v_raw_opacities)
    state.backward_called = True


__all__ = [
    "ExplicitGaussianState",
    "explicit_backward",
    "explicit_rasterize",
    "explicit_voxelize",
    "prepare_explicit_gaussians",
]
