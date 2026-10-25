/**
 * CUDA kernel launch declarations for world-to-volume mapping.
 */

#ifndef R2SPLAT_WORLD_TO_VOLUME_MAPPING_H
#define R2SPLAT_WORLD_TO_VOLUME_MAPPING_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
void launch_world_to_volume_mapping_fwd(
    const S* __restrict__ means,
    const S* __restrict__ covars,
    const S* __restrict__ max_scales,
    const S* __restrict__ ranges,
    const S* __restrict__ centers,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    S radius_clip,
    S* __restrict__ means3d_v,
    S* __restrict__ conics,
    int32_t* __restrict__ radii,
    uint32_t N,
    cudaStream_t stream);

template <typename S>
void launch_world_to_volume_mapping_bwd(
    const S* __restrict__ means,
    const S* __restrict__ covars,
    const S* __restrict__ ranges,
    const S* __restrict__ centers,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    const S* __restrict__ v_means3d_v,
    const S* __restrict__ v_conics,
    S* __restrict__ v_means,
    S* __restrict__ v_covars,
    uint32_t N,
    cudaStream_t stream);

} // namespace r2splat

#endif // R2SPLAT_WORLD_TO_VOLUME_MAPPING_H
