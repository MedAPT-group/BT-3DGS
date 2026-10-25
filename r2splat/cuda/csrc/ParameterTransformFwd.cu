/**
 * @file ParameterTransformFwd.cu
 * @brief Forward CUDA kernel: (quats, scale_ori) -> covariance matrix
 *
 * Transforms unit quaternions and raw scale parameters into world-space
 * 3D covariance matrices for Gaussian splatting.
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * Given:
 *   q_raw = (w, x, y, z)  -- unconstrained quaternion parameter
 *   scale_ori = (so_0, so_1, so_2)  -- raw scale parameters
 *   scale_min, scale_max  -- activation range
 *
 * Step 1: Scale activation via sigmoid
 *   sig_i = 1 / (1 + exp(-so_i))
 *   s_i = sig_i * (scale_max - scale_min) + scale_min
 *
 * Step 2: Normalize q_raw and convert it to a rotation matrix
 *   x2 = x*x, y2 = y*y, z2 = z*z
 *   xy = x*y, xz = x*z, yz = y*z
 *   wx = w*x, wy = w*y, wz = w*z
 *
 *   R = [[1-2(y2+z2),  2(xy-wz),   2(xz+wy)],
 *        [2(xy+wz),    1-2(x2+z2), 2(yz-wx)],
 *        [2(xz-wy),    2(yz+wx),   1-2(x2+y2)]]
 *
 * Step 3: Build M = R * S, then C = M * M^T
 *   C_ij = sum_k M_ik * M_jk = sum_k R_ik * s_k * R_jk * s_k
 *        = sum_k R_ik * R_jk * s_k^2
 *
 * Step 4: Store as upper-triangular (C00, C01, C02, C11, C12, C22)
 *
 * Each thread processes one Gaussian.
 *
 * Author: Dy164
 * Date: 2026-02-20
 */

#include "ParameterTransform.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename S>
__global__ void parameter_transform_fwd_kernel(
    const S* __restrict__ quats,       // [N, 4]  (w, x, y, z)
    const S* __restrict__ scale_ori,   // [N, 3]
    const S scale_min,
    const S scale_max,
    S* __restrict__ covars,            // [N, 6]  upper-tri (C00,C01,C02,C11,C12,C22)
    S* __restrict__ max_scales,        // [N]     max(s0, s1, s2)
    uint32_t N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Normalize the raw quaternion, matching torch.nn.functional.normalize ----
    const S rw = quats[idx * 4 + 0];
    const S rx = quats[idx * 4 + 1];
    const S ry = quats[idx * 4 + 2];
    const S rz = quats[idx * 4 + 3];
    const S qnorm = sqrt(rw * rw + rx * rx + ry * ry + rz * rz);
    const S qdenom = max(qnorm, static_cast<S>(1e-12));
    const S inv_qnorm = static_cast<S>(1) / qdenom;
    const S w = rw * inv_qnorm;
    const S x = rx * inv_qnorm;
    const S y = ry * inv_qnorm;
    const S z = rz * inv_qnorm;

    // ---- Read and activate scale ----
    const S range = scale_max - scale_min;
    S s[3];
    for (int i = 0; i < 3; ++i) {
        const S so = scale_ori[idx * 3 + i];
        const S sig = static_cast<S>(1) / (static_cast<S>(1) + exp(-so));
        s[i] = sig * range + scale_min;
    }

    // ---- Write max_scales (non-differentiable) ----
    max_scales[idx] = max(s[0], max(s[1], s[2]));

    // ---- Quaternion to rotation matrix ----
    // Precompute products
    const S x2 = x * x, y2 = y * y, z2 = z * z;
    const S xy = x * y, xz = x * z, yz = y * z;
    const S wx = w * x, wy = w * y, wz = w * z;

    // R is stored as R[col][row] to match column-major convention,
    // but we compute row-major elements directly for clarity.
    // R[i][j] = element at row i, col j
    const S R00 = static_cast<S>(1) - static_cast<S>(2) * (y2 + z2);
    const S R01 = static_cast<S>(2) * (xy - wz);
    const S R02 = static_cast<S>(2) * (xz + wy);
    const S R10 = static_cast<S>(2) * (xy + wz);
    const S R11 = static_cast<S>(1) - static_cast<S>(2) * (x2 + z2);
    const S R12 = static_cast<S>(2) * (yz - wx);
    const S R20 = static_cast<S>(2) * (xz - wy);
    const S R21 = static_cast<S>(2) * (yz + wx);
    const S R22 = static_cast<S>(1) - static_cast<S>(2) * (x2 + y2);

    // ---- Compute covariance C_ij = sum_k R_ik * R_jk * s_k^2 ----
    const S s0_sq = s[0] * s[0];
    const S s1_sq = s[1] * s[1];
    const S s2_sq = s[2] * s[2];

    const S C00 = R00 * R00 * s0_sq + R01 * R01 * s1_sq + R02 * R02 * s2_sq;
    const S C01 = R00 * R10 * s0_sq + R01 * R11 * s1_sq + R02 * R12 * s2_sq;
    const S C02 = R00 * R20 * s0_sq + R01 * R21 * s1_sq + R02 * R22 * s2_sq;
    const S C11 = R10 * R10 * s0_sq + R11 * R11 * s1_sq + R12 * R12 * s2_sq;
    const S C12 = R10 * R20 * s0_sq + R11 * R21 * s1_sq + R12 * R22 * s2_sq;
    const S C22 = R20 * R20 * s0_sq + R21 * R21 * s1_sq + R22 * R22 * s2_sq;

    // ---- Write upper-triangular output ----
    S* out = covars + idx * 6;
    out[0] = C00;
    out[1] = C01;
    out[2] = C02;
    out[3] = C11;
    out[4] = C12;
    out[5] = C22;
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_parameter_transform_fwd(
    const S* __restrict__ quats,
    const S* __restrict__ scale_ori,
    S scale_min,
    S scale_max,
    S* __restrict__ covars,
    S* __restrict__ max_scales,
    uint32_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    parameter_transform_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        quats, scale_ori, scale_min, scale_max, covars, max_scales, N
    );
}

// Explicit template instantiations
template void launch_parameter_transform_fwd<float>(
    const float*, const float*, float, float, float*, float*, uint32_t, cudaStream_t
);
template void launch_parameter_transform_fwd<double>(
    const double*, const double*, double, double, double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
