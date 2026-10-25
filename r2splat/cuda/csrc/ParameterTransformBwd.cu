/**
 * @file ParameterTransformBwd.cu
 * @brief First-order backward CUDA kernel: gradients of covar w.r.t. (quats, scale_ori)
 *
 * Given downstream gradient dL/d(covars) [N, 6], computes:
 *   dL/d(quats) [N, 4]      via chain rule through R -> M -> C
 *   dL/d(scale_ori) [N, 3]  via chain rule through sigmoid -> s -> M -> C
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * Chain: scale_ori -> s (sigmoid) -> S (diag) -> M = R*S -> C = M*M^T
 *
 * Step 1: v_C -> v_M
 *   C = M * M^T, so (using matrix calculus for symmetric output):
 *   dL/dM = (v_C + v_C^T) * M
 *
 *   Since C is symmetric and we store upper-tri, v_C as a full 3x3 matrix is:
 *     v_C = [[v0, v1, v2],
 *            [v1, v3, v4],
 *            [v2, v4, v5]]
 *   (off-diagonal v_C elements already have factor 1, not 1/2, because
 *    the upper-tri gradient accounts for both symmetric positions.)
 *   Actually v_C is already symmetric, so v_C + v_C^T = 2*v_C.
 *   Thus v_M = 2 * v_C * M.
 *
 * Step 2: v_M -> v_R
 *   M = R * S, so v_R = v_M * S^T = v_M * S  (S is diagonal)
 *   v_R[i][j] = v_M[i][j] * s[j]
 *
 * Step 3: v_R -> v_quat  (Euclidean gradient, no projection for normalized q)
 *   Standard quaternion-to-rotation VJP formulas.
 *
 * Step 4: v_M -> v_s -> v_scale_ori
 *   M_ij = R_ij * s_j, so:
 *   v_s[j] = sum_i R_ij * v_M[i][j]
 *
 *   Then apply sigmoid chain rule:
 *   v_scale_ori[j] = v_s[j] * sig_j * (1 - sig_j) * (scale_max - scale_min)
 *
 * Note on v_C symmetry:
 *   The downstream v_covars stores 6 elements for the upper triangle.
 *   When we form the full 3x3 matrix, off-diagonal elements appear once
 *   in the stored gradient. Since C01 = C10, a perturbation in C01
 *   contributes to both positions, so v_C as a full matrix has the
 *   off-diagonal entries equal (not halved). Thus:
 *   v_C_full = [[v0, v1, v2], [v1, v3, v4], [v2, v4, v5]]
 *   v_C + v_C^T = 2 * v_C_full
 *
 * Author: Dy164
 * Date: 2026-02-20
 */

#include "ParameterTransform.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// First-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void parameter_transform_bwd_kernel(
    const S* __restrict__ quats,       // [N, 4]  (w, x, y, z)
    const S* __restrict__ scale_ori,   // [N, 3]
    const S scale_min,
    const S scale_max,
    const S* __restrict__ v_covars,    // [N, 6]  dL/d(C00,C01,C02,C11,C12,C22)
    S* __restrict__ v_quats,           // [N, 4]
    S* __restrict__ v_scale_ori,       // [N, 3]
    uint32_t N
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Normalize the raw quaternion, matching the forward transform ----
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
    S s[3], sig[3];
    for (int i = 0; i < 3; ++i) {
        const S so = scale_ori[idx * 3 + i];
        sig[i] = static_cast<S>(1) / (static_cast<S>(1) + exp(-so));
        s[i] = sig[i] * range + scale_min;
    }

    // ---- Quaternion to rotation matrix (row-major: R[row][col]) ----
    const S x2 = x * x, y2 = y * y, z2 = z * z;
    const S xy = x * y, xz = x * z, yz = y * z;
    const S wx = w * x, wy = w * y, wz = w * z;

    const S R00 = static_cast<S>(1) - static_cast<S>(2) * (y2 + z2);
    const S R01 = static_cast<S>(2) * (xy - wz);
    const S R02 = static_cast<S>(2) * (xz + wy);
    const S R10 = static_cast<S>(2) * (xy + wz);
    const S R11 = static_cast<S>(1) - static_cast<S>(2) * (x2 + z2);
    const S R12 = static_cast<S>(2) * (yz - wx);
    const S R20 = static_cast<S>(2) * (xz - wy);
    const S R21 = static_cast<S>(2) * (yz + wx);
    const S R22 = static_cast<S>(1) - static_cast<S>(2) * (x2 + y2);

    // ---- Read downstream gradient (upper-tri of symmetric 3x3) ----
    const S* vc = v_covars + idx * 6;
    const S v0 = vc[0];  // dL/dC00 (diagonal)
    const S v1 = vc[1] * static_cast<S>(0.5);  // dL/dC01 (off-diagonal, halved)
    const S v2 = vc[2] * static_cast<S>(0.5);  // dL/dC02 (off-diagonal, halved)
    const S v3 = vc[3];  // dL/dC11 (diagonal)
    const S v4 = vc[4] * static_cast<S>(0.5);  // dL/dC12 (off-diagonal, halved)
    const S v5 = vc[5];  // dL/dC22 (diagonal)

    // ---- Compute v_M = 2 * v_C_half * M ----
    // The covariance C = M * M^T is symmetric, stored as upper-tri [C00,C01,C02,C11,C12,C22].
    // v_covars are gradients w.r.t. these 6 stored elements.
    // When constructing the full symmetric gradient matrix, off-diagonal elements
    // must be halved to avoid double-counting:
    //   v_C_half = [[v0, v1/2, v2/2], [v1/2, v3, v4/2], [v2/2, v4/2, v5]]
    // Then: dL/dM = 2 * v_C_half * M
    //
    // v_C_half = [[v0, v1, v2], [v1, v3, v4], [v2, v4, v5]]  (v1,v2,v4 already halved above)
    // M[i][j] = R[i][j] * s[j]
    //
    // (2 * v_C_half * M)[i][j] = 2 * sum_k v_C_half[i][k] * M[k][j]
    //                           = 2 * sum_k v_C_half[i][k] * R[k][j] * s[j]

    // v_M[i][j] for i=0..2, j=0..2
    S vM[3][3];
    // Row 0 of v_M
    vM[0][0] = static_cast<S>(2) * (v0 * R00 + v1 * R10 + v2 * R20) * s[0];
    vM[0][1] = static_cast<S>(2) * (v0 * R01 + v1 * R11 + v2 * R21) * s[1];
    vM[0][2] = static_cast<S>(2) * (v0 * R02 + v1 * R12 + v2 * R22) * s[2];
    // Row 1 of v_M
    vM[1][0] = static_cast<S>(2) * (v1 * R00 + v3 * R10 + v4 * R20) * s[0];
    vM[1][1] = static_cast<S>(2) * (v1 * R01 + v3 * R11 + v4 * R21) * s[1];
    vM[1][2] = static_cast<S>(2) * (v1 * R02 + v3 * R12 + v4 * R22) * s[2];
    // Row 2 of v_M
    vM[2][0] = static_cast<S>(2) * (v2 * R00 + v4 * R10 + v5 * R20) * s[0];
    vM[2][1] = static_cast<S>(2) * (v2 * R01 + v4 * R11 + v5 * R21) * s[1];
    vM[2][2] = static_cast<S>(2) * (v2 * R02 + v4 * R12 + v5 * R22) * s[2];

    // ---- v_R = v_M * S (column-wise scaling by s[j]) ----
    // C = M * M^T where M = R * S
    // dL/dM = 2 * v_C_half * M = 2 * v_C_half * R * S
    // dL/dR = dL/dM * S^T = dL/dM * S (S diagonal)
    //       = 2 * v_C_half * R * S^2
    //
    // So v_R[i][j] = 2 * sum_k v_C_half[i][k] * R[k][j] * s[j]^2

    S vR[3][3];
    vR[0][0] = static_cast<S>(2) * (v0 * R00 + v1 * R10 + v2 * R20) * s[0] * s[0];
    vR[0][1] = static_cast<S>(2) * (v0 * R01 + v1 * R11 + v2 * R21) * s[1] * s[1];
    vR[0][2] = static_cast<S>(2) * (v0 * R02 + v1 * R12 + v2 * R22) * s[2] * s[2];
    vR[1][0] = static_cast<S>(2) * (v1 * R00 + v3 * R10 + v4 * R20) * s[0] * s[0];
    vR[1][1] = static_cast<S>(2) * (v1 * R01 + v3 * R11 + v4 * R21) * s[1] * s[1];
    vR[1][2] = static_cast<S>(2) * (v1 * R02 + v3 * R12 + v4 * R22) * s[2] * s[2];
    vR[2][0] = static_cast<S>(2) * (v2 * R00 + v4 * R10 + v5 * R20) * s[0] * s[0];
    vR[2][1] = static_cast<S>(2) * (v2 * R01 + v4 * R11 + v5 * R21) * s[1] * s[1];
    vR[2][2] = static_cast<S>(2) * (v2 * R02 + v4 * R12 + v5 * R22) * s[2] * s[2];

    // ---- v_quat from v_R (Euclidean VJP for unit quaternion) ----
    // Using standard formula: v_q_a = sum_{i,j} vR[i][j] * dR[i][j]/dq_a
    //
    // dR/dw = [[0,-2z,2y],[2z,0,-2x],[-2y,2x,0]]
    // => v_w = 2*(x*(vR21-vR12) + y*(vR02-vR20) + z*(vR10-vR01))
    //
    // dR/dx = [[0,2y,2z],[2y,-4x,-2w],[2z,2w,-4x]]
    // => v_x = 2*(-2x*(vR11+vR22) + y*(vR01+vR10) + z*(vR02+vR20) + w*(vR21-vR12))
    //
    // dR/dy = [[-4y,2x,2w],[2x,0,2z],[-2w,2z,-4y]]
    // => v_y = 2*(x*(vR01+vR10) - 2y*(vR00+vR22) + z*(vR12+vR21) + w*(vR02-vR20))
    //
    // dR/dz = [[-4z,-2w,2x],[2w,-4z,2y],[2x,2y,0]]
    // => v_z = 2*(x*(vR02+vR20) + y*(vR12+vR21) - 2z*(vR00+vR11) + w*(vR10-vR01))

    const S vw = static_cast<S>(2) * (
        x * (vR[2][1] - vR[1][2]) +
        y * (vR[0][2] - vR[2][0]) +
        z * (vR[1][0] - vR[0][1])
    );
    const S vx = static_cast<S>(2) * (
        static_cast<S>(-2) * x * (vR[1][1] + vR[2][2]) +
        y * (vR[0][1] + vR[1][0]) +
        z * (vR[0][2] + vR[2][0]) +
        w * (vR[2][1] - vR[1][2])
    );
    const S vy = static_cast<S>(2) * (
        x * (vR[0][1] + vR[1][0]) +
        static_cast<S>(-2) * y * (vR[0][0] + vR[2][2]) +
        z * (vR[1][2] + vR[2][1]) +
        w * (vR[0][2] - vR[2][0])
    );
    const S vz = static_cast<S>(2) * (
        x * (vR[0][2] + vR[2][0]) +
        y * (vR[1][2] + vR[2][1]) +
        static_cast<S>(-2) * z * (vR[0][0] + vR[1][1]) +
        w * (vR[1][0] - vR[0][1])
    );

    // ---- v_s from v_M ----
    // M_ij = R_ij * s_j, so v_s[j] = sum_i R_ij * v_M_ij
    // v_M_ij = 2 * sum_k vC_full[i][k] * R[k][j] * s[j]
    // So v_s[j] = sum_i R_ij * 2 * sum_k vC_full[i][k] * R[k][j] * s[j]
    //           = 2 * s[j] * sum_i sum_k R_ij * R_kj * vC_full[i][k]
    // Note: sum_i sum_k R_ij * R_kj * vC_full[i][k] = sum_i R_ij * (sum_k vC_full[i][k] * R_kj)

    S vs[3];
    vs[0] = R00 * vM[0][0] + R10 * vM[1][0] + R20 * vM[2][0];
    vs[1] = R01 * vM[0][1] + R11 * vM[1][1] + R21 * vM[2][1];
    vs[2] = R02 * vM[0][2] + R12 * vM[1][2] + R22 * vM[2][2];

    // ---- v_scale_ori via sigmoid chain rule ----
    // v_scale_ori[j] = v_s[j] * sig[j] * (1 - sig[j]) * range
    S* v_so = v_scale_ori + idx * 3;
    for (int j = 0; j < 3; ++j) {
        v_so[j] = vs[j] * sig[j] * (static_cast<S>(1) - sig[j]) * range;
    }

    // ---- Backpropagate through q = q_raw / max(||q_raw||, eps) ----
    S* vq = v_quats + idx * 4;
    if (qnorm > static_cast<S>(1e-12)) {
        const S dot = w * vw + x * vx + y * vy + z * vz;
        vq[0] = (vw - w * dot) * inv_qnorm;
        vq[1] = (vx - x * dot) * inv_qnorm;
        vq[2] = (vy - y * dot) * inv_qnorm;
        vq[3] = (vz - z * dot) * inv_qnorm;
    } else {
        vq[0] = vw * inv_qnorm;
        vq[1] = vx * inv_qnorm;
        vq[2] = vy * inv_qnorm;
        vq[3] = vz * inv_qnorm;
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_parameter_transform_bwd(
    const S* __restrict__ quats,
    const S* __restrict__ scale_ori,
    S scale_min,
    S scale_max,
    const S* __restrict__ v_covars,
    S* __restrict__ v_quats,
    S* __restrict__ v_scale_ori,
    uint32_t N,
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    parameter_transform_bwd_kernel<S><<<blocks, threads, 0, stream>>>(
        quats, scale_ori, scale_min, scale_max,
        v_covars, v_quats, v_scale_ori, N
    );
}

// Explicit template instantiations
template void launch_parameter_transform_bwd<float>(
    const float*, const float*, float, float,
    const float*, float*, float*, uint32_t, cudaStream_t
);
template void launch_parameter_transform_bwd<double>(
    const double*, const double*, double, double,
    const double*, double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
