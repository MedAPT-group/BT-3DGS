/**
 * CUDA kernel launch declarations for CT projection rasterization.
 */

#ifndef R2SPLAT_RASTERIZE_TO_PIXELS_H
#define R2SPLAT_RASTERIZE_TO_PIXELS_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
void launch_rasterize_to_pixels_fwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_rasterize_to_pixels_bwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means2d_abs,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);

template <typename S>
void launch_rasterize_to_pixels_bwd_per_gaussian(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means2d_abs,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream);


} // namespace r2splat

#endif // R2SPLAT_RASTERIZE_TO_PIXELS_H

