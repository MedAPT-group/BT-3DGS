"""Thin Python wrappers for the installed CUDA extension."""

from __future__ import annotations

from typing import Callable


def _make_cuda_func(name: str) -> Callable:
    def call_cuda(*args, **kwargs):
        from ._backend import _C

        return getattr(_C, name)(*args, **kwargs)

    call_cuda.__name__ = name
    return call_cuda


isect_tiles = _make_cuda_func("isect_tiles")
parameter_transform_fwd = _make_cuda_func("parameter_transform_fwd")
parameter_transform_bwd = _make_cuda_func("parameter_transform_bwd")
isect_offset_encode = _make_cuda_func("isect_offset_encode")
isect_tiles_3d = _make_cuda_func("isect_tiles_3d")
isect_offset_encode_3d = _make_cuda_func("isect_offset_encode_3d")
conics_mu_fwd = _make_cuda_func("conics_mu_fwd")
conics_mu_bwd = _make_cuda_func("conics_mu_bwd")
world_to_image_projection_fwd = _make_cuda_func("world_to_image_projection_fwd")
world_to_image_projection_bwd = _make_cuda_func("world_to_image_projection_bwd")
world_to_volume_mapping_fwd = _make_cuda_func("world_to_volume_mapping_fwd")
world_to_volume_mapping_bwd = _make_cuda_func("world_to_volume_mapping_bwd")
rasterize_to_pixels_fwd = _make_cuda_func("rasterize_to_pixels_fwd")
rasterize_to_pixels_bwd = _make_cuda_func("rasterize_to_pixels_bwd")
rasterize_to_pixels_bwd_per_gaussian = _make_cuda_func(
    "rasterize_to_pixels_bwd_per_gaussian"
)
voxelize_to_voxels_fwd = _make_cuda_func("voxelize_to_voxels_fwd")
voxelize_to_voxels_bwd = _make_cuda_func("voxelize_to_voxels_bwd")
voxelize_to_voxels_bwd_per_gaussian = _make_cuda_func(
    "voxelize_to_voxels_bwd_per_gaussian"
)


def is_cuda_extension_available() -> bool:
    try:
        from ._backend import is_cuda_available

        return is_cuda_available()
    except ImportError:
        return False


__all__ = [name for name in globals() if not name.startswith("_")]
