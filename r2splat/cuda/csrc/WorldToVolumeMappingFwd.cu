/**
 * @file WorldToVolumeMappingFwd.cu
 * @brief Forward CUDA kernel: world-to-volume mapping
 *
 * Transforms world-space Gaussian parameters into voxel (volume) space.
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * Given:
 *   p = (px, py, pz)       -- world-space Gaussian center
 *   Sigma = 3x3 symmetric  -- world-space covariance (upper-tri: s0..s5)
 *   max_scale              -- max activated scale (for radii only)
 *   ranges = (rx, ry, rz)  -- physical extent of volume in each axis
 *   centers = (cx, cy, cz) -- physical center of volume
 *   dim2, dim1, dim0       -- voxel grid dimensions
 *   eps3d                  -- blur epsilon
 *   radius_clip            -- radius multiplier
 *
 * Step 1: Compute scaling factors (R is diagonal)
 *   r = (dim2/rx, dim1/ry, dim0/rz)
 *
 * Step 2: Affine transform for means
 *   means3d_v[k] = (p[k] - center[k]) * r[k] + dim[k]/2
 *   where k: 0→x(dim2), 1→y(dim1), 2→z(dim0)
 *
 * Step 3: Covariance transform (R diagonal)
 *   Sigma_v[i][j] = r[i] * r[j] * Sigma[i][j]
 *
 * Step 4: Add blur
 *   Sigma_v[i][i] += eps3d
 *
 * Step 5: 3x3 symmetric matrix inversion → conics
 *   conics = Sigma_v^{-1}
 *
 * Step 6: Axis-aligned radii (non-differentiable)
 *   radii[k] = ceil(radius_clip * sqrt(Sigma_v[k,k]))
 *
 * Each thread processes one Gaussian.
 *
 * Author: Dy164
 * Date: 2026-02-21
 */

#include "WorldToVolumeMapping.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Helper: Invert 3x3 symmetric matrix from upper-triangular storage
// ============================================================================
template <typename S>
__device__ __forceinline__ bool invert_sym3x3(
    S a00, S a01, S a02, S a11, S a12, S a22,
    S& i00, S& i01, S& i02, S& i11, S& i12, S& i22
) {
    // Cofactors
    const S c00 = a11 * a22 - a12 * a12;
    const S c01 = a02 * a12 - a01 * a22;
    const S c02 = a01 * a12 - a02 * a11;
    const S c11 = a00 * a22 - a02 * a02;
    const S c12 = a01 * a02 - a00 * a12;
    const S c22 = a00 * a11 - a01 * a01;

    const S det = a00 * c00 + a01 * c01 + a02 * c02;

    if (abs(det) < static_cast<S>(1e-12)) {
        i00 = i01 = i02 = i11 = i12 = i22 = static_cast<S>(0);
        return false;
    }

    const S inv_det = static_cast<S>(1) / det;
    i00 = c00 * inv_det;
    i01 = c01 * inv_det;
    i02 = c02 * inv_det;
    i11 = c11 * inv_det;
    i12 = c12 * inv_det;
    i22 = c22 * inv_det;
    return true;
}

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_volume_mapping_fwd_kernel(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ max_scales,  // [N]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    S radius_clip,
    S* __restrict__ means3d_v,         // [N, 3]
    S* __restrict__ conics_out,        // [N, 6]
    int32_t* __restrict__ radii,       // [N, 3]
    uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Read scaling factors ----
    const S rx = ranges[0], ry = ranges[1], rz = ranges[2];
    const S cx = centers[0], cy = centers[1], cz = centers[2];
    const S D2 = static_cast<S>(dim2);
    const S D1 = static_cast<S>(dim1);
    const S D0 = static_cast<S>(dim0);

    // r[k] = dim[k] / range[k]
    // means stored as (x, y, z) = (dim2, dim1, dim0)
    const S r0 = D2 / rx;  // x scaling
    const S r1 = D1 / ry;  // y scaling
    const S r2 = D0 / rz;  // z scaling

    // ---- Read world-space Gaussian ----
    const S px = means[idx * 3 + 0];
    const S py = means[idx * 3 + 1];
    const S pz = means[idx * 3 + 2];

    // ---- Step 2: Affine transform ----
    means3d_v[idx * 3 + 0] = (px - cx) * r0 + D2 * static_cast<S>(0.5);
    means3d_v[idx * 3 + 1] = (py - cy) * r1 + D1 * static_cast<S>(0.5);
    means3d_v[idx * 3 + 2] = (pz - cz) * r2 + D0 * static_cast<S>(0.5);

    // ---- Read world covariance (upper-tri) ----
    const S* s = covars + idx * 6;
    const S s00 = s[0], s01 = s[1], s02 = s[2];
    const S s11 = s[3], s12 = s[4], s22 = s[5];

    // ---- Step 3: Covariance transform (R diagonal) ----
    // Sigma_v[i][j] = r[i] * r[j] * Sigma[i][j]
    S Sv00 = r0 * r0 * s00;
    S Sv01 = r0 * r1 * s01;
    S Sv02 = r0 * r2 * s02;
    S Sv11 = r1 * r1 * s11;
    S Sv12 = r1 * r2 * s12;
    S Sv22 = r2 * r2 * s22;

    // ---- Step 4: Add blur ----
    Sv00 += eps3d;
    Sv11 += eps3d;
    Sv22 += eps3d;

    // ---- Step 5: Invert to get conics ----
    S c00, c01, c02, c11, c12, c22;
    bool ok = invert_sym3x3(Sv00, Sv01, Sv02, Sv11, Sv12, Sv22,
                            c00, c01, c02, c11, c12, c22);

    S* out = conics_out + idx * 6;
    if (!ok) {
        for (int i = 0; i < 6; ++i) out[i] = static_cast<S>(0);
        radii[idx * 3 + 0] = 0;
        radii[idx * 3 + 1] = 0;
        radii[idx * 3 + 2] = 0;
        return;
    }

    out[0] = c00;
    out[1] = c01;
    out[2] = c02;
    out[3] = c11;
    out[4] = c12;
    out[5] = c22;

    // ---- Step 6: Compute tight axis-aligned radii (non-differentiable) ----
    const S radius_x = ceil(radius_clip * sqrt(max(static_cast<S>(0), Sv00)));
    const S radius_y = ceil(radius_clip * sqrt(max(static_cast<S>(0), Sv11)));
    const S radius_z = ceil(radius_clip * sqrt(max(static_cast<S>(0), Sv22)));
    if (!isfinite(radius_x) || !isfinite(radius_y) || !isfinite(radius_z) ||
        radius_x <= static_cast<S>(0) ||
        radius_y <= static_cast<S>(0) ||
        radius_z <= static_cast<S>(0)) {
        radii[idx * 3 + 0] = 0;
        radii[idx * 3 + 1] = 0;
        radii[idx * 3 + 2] = 0;
        return;
    }

    radii[idx * 3 + 0] = static_cast<int32_t>(radius_x);
    radii[idx * 3 + 1] = static_cast<int32_t>(radius_y);
    radii[idx * 3 + 2] = static_cast<int32_t>(radius_z);
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

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
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    world_to_volume_mapping_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, max_scales, ranges, centers,
        dim2, dim1, dim0, eps3d, radius_clip,
        means3d_v, conics, radii, N
    );
}

// Explicit template instantiations
template void launch_world_to_volume_mapping_fwd<float>(
    const float*, const float*, const float*, const float*, const float*,
    uint32_t, uint32_t, uint32_t, float, float,
    float*, float*, int32_t*, uint32_t, cudaStream_t
);
template void launch_world_to_volume_mapping_fwd<double>(
    const double*, const double*, const double*, const double*, const double*,
    uint32_t, uint32_t, uint32_t, double, double,
    double*, double*, int32_t*, uint32_t, cudaStream_t
);

} // namespace r2splat
