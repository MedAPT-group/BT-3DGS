/**
 * CUDA kernel launch declarations for world-to-image projection.
 */

#ifndef R2SPLAT_WORLD_TO_IMAGE_PROJECTION_H
#define R2SPLAT_WORLD_TO_IMAGE_PROJECTION_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
void launch_world_to_image_projection_fwd(
    const S* __restrict__ means,
    const S* __restrict__ covars,
    const S* __restrict__ viewmats,
    const S* __restrict__ Ks,
    uint32_t dim1,
    uint32_t dim0,
    S eps_blur,
    S radius_clip,
    S* __restrict__ means2d,
    S* __restrict__ cov2d_3x3,
    int32_t* __restrict__ radii,
    uint32_t C,
    uint32_t N,
    cudaStream_t stream);

template <typename S>
void launch_world_to_image_projection_bwd(
    const S* __restrict__ means,
    const S* __restrict__ covars,
    const S* __restrict__ viewmats,
    const S* __restrict__ Ks,
    uint32_t dim1,
    uint32_t dim0,
    const S* __restrict__ v_means2d,
    const S* __restrict__ v_cov2d_3x3,
    S* __restrict__ v_means,
    S* __restrict__ v_covars,
    uint32_t C,
    uint32_t N,
    cudaStream_t stream);

} // namespace r2splat

#endif // R2SPLAT_WORLD_TO_IMAGE_PROJECTION_H
