/**
 * @file WorldToVolumeMappingBwd.cu
 * @brief First-order backward CUDA kernel: world-to-volume mapping
 *
 * Computes VJP (gradients) through the world-to-volume mapping.
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * Forward:
 *   means3d_v[k] = (p[k] - center[k]) * r[k] + dim[k]/2
 *   Sigma_v = R * Sigma_w * R^T + eps*I    (R diagonal)
 *   conics = Sigma_v^{-1}
 *
 * Gradient path 1: means3d_v → means (affine, simple scaling)
 *   v_means[k] = v_means3d_v[k] * r[k]
 *
 * Gradient path 2: conics → Sigma_v → covars
 *   conics_utri stores ONLY the upper triangle of P = Sigma_v^{-1}.
 *   The effective dL/dP_full is UPPER TRIANGULAR (not symmetric!):
 *     V_P_upper[i][j] = v_conics[utri(i,j)]  for i<=j
 *     V_P_upper[i][j] = 0                     for i>j
 *
 *   Standard matrix inverse VJP:
 *     V_Sv_full = -P @ V_P_upper @ P     (result is NOT symmetric)
 *
 *   Since covars_utri[flat] for off-diag (i<j) contributes to both
 *   Sv[i][j] and Sv[j][i]:
 *     v_covars[flat] = V_Sv[i][i] * r[i]^2                  for i==j
 *     v_covars[flat] = (V_Sv[i][j] + V_Sv[j][i]) * r[i]*r[j]  for i<j
 *
 * Note: blur eps is constant, so d(Sigma_v)/d(eps) doesn't contribute
 *       to covars gradient.
 *
 * Author: Dy164
 * Date: 2026-02-21
 */

#include "WorldToVolumeMapping.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Helper: Invert 3x3 symmetric matrix
// ============================================================================
template <typename S>
__device__ __forceinline__ bool invert_sym3x3(
    S a00, S a01, S a02, S a11, S a12, S a22,
    S& i00, S& i01, S& i02, S& i11, S& i12, S& i22
) {
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
// Backward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_volume_mapping_bwd_kernel(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ ranges,      // [3]
    const S* __restrict__ centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    S eps3d,
    const S* __restrict__ v_means3d_v, // [N, 3]
    const S* __restrict__ v_conics,    // [N, 6]
    S* __restrict__ v_means,           // [N, 3]
    S* __restrict__ v_covars,          // [N, 6]
    uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // ---- Compute scaling factors ----
    const S rx = ranges[0], ry = ranges[1], rz = ranges[2];
    const S D2 = static_cast<S>(dim2);
    const S D1 = static_cast<S>(dim1);
    const S D0 = static_cast<S>(dim0);
    const S r0 = D2 / rx;
    const S r1 = D1 / ry;
    const S r2 = D0 / rz;

    // ---- Gradient path 1: means3d_v -> means ----
    // means3d_v = (means - center) * r + dim / 2, so its VJP scales by r.
    v_means[idx * 3 + 0] = v_means3d_v[idx * 3 + 0] * r0;
    v_means[idx * 3 + 1] = v_means3d_v[idx * 3 + 1] * r1;
    v_means[idx * 3 + 2] = v_means3d_v[idx * 3 + 2] * r2;

    // ---- Recompute Sigma_v and conics for gradient path 2 ----
    const S* s = covars + idx * 6;
    const S s00 = s[0], s01 = s[1], s02 = s[2];
    const S s11 = s[3], s12 = s[4], s22 = s[5];

    S Sv00 = r0 * r0 * s00 + eps3d;
    S Sv01 = r0 * r1 * s01;
    S Sv02 = r0 * r2 * s02;
    S Sv11 = r1 * r1 * s11 + eps3d;
    S Sv12 = r1 * r2 * s12;
    S Sv22 = r2 * r2 * s22 + eps3d;

    // Compute conics = Sigma_v^{-1}
    S P00, P01, P02, P11, P12, P22;
    bool ok = invert_sym3x3(Sv00, Sv01, Sv02, Sv11, Sv12, Sv22,
                            P00, P01, P02, P11, P12, P22);
    if (!ok) {
        for (int i = 0; i < 6; ++i) v_covars[idx * 6 + i] = static_cast<S>(0);
        return;
    }

    // ---- Gradient path 2: conics → Sigma_v → covars ----
    //
    // Key insight: conics_utri stores ONLY the upper triangle of P = Sv^{-1}.
    // So v_conics is dL/d(conics_utri), NOT dL/d(P_full).
    // The correct V_P_full is UPPER TRIANGULAR (lower triangle = 0),
    // because conics_utri[flat] = P[i][j] for i<=j only.
    //
    // Step 1: Build V_P as upper-triangular matrix (NOT symmetric!)
    // Step 2: v_Sv_full = -P @ V_P_upper @ P   (result is NOT symmetric)
    // Step 3: Since covars_utri[flat] for off-diag (i<j) contributes to
    //         both Sv[i][j] and Sv[j][i], we need:
    //           v_covars[flat] = (v_Sv[i][j] + v_Sv[j][i]) * r[i]*r[j]  for i<j
    //           v_covars[flat] = v_Sv[i][i] * r[i]*r[i]                  for i==j

    const S vc00 = v_conics[idx * 6 + 0];
    const S vc01 = v_conics[idx * 6 + 1];
    const S vc02 = v_conics[idx * 6 + 2];
    const S vc11 = v_conics[idx * 6 + 3];
    const S vc12 = v_conics[idx * 6 + 4];
    const S vc22 = v_conics[idx * 6 + 5];

    // V_P_upper (upper triangular, lower = 0):
    //   [ vc00  vc01  vc02 ]
    //   [  0    vc11  vc12 ]
    //   [  0     0    vc22 ]

    // Compute T = V_P_upper @ P
    // T[i][j] = sum_k V_upper[i][k] * P[k][j]
    // Row 0: V_upper row = (vc00, vc01, vc02)
    const S T00 = vc00*P00 + vc01*P01 + vc02*P02;
    const S T01 = vc00*P01 + vc01*P11 + vc02*P12;
    const S T02 = vc00*P02 + vc01*P12 + vc02*P22;
    // Row 1: V_upper row = (0, vc11, vc12)
    const S T10 = vc11*P01 + vc12*P02;
    const S T11 = vc11*P11 + vc12*P12;
    const S T12 = vc11*P12 + vc12*P22;
    // Row 2: V_upper row = (0, 0, vc22)
    const S T20 = vc22*P02;
    const S T21 = vc22*P12;
    const S T22 = vc22*P22;

    // Compute v_Sv_full = -P @ T  (full 3x3, NOT symmetric)
    // v_Sv_full[i][j] = - sum_k P[i][k] * T[k][j]
    const S vSv00 = -(P00*T00 + P01*T10 + P02*T20);
    const S vSv01 = -(P00*T01 + P01*T11 + P02*T21);
    const S vSv02 = -(P00*T02 + P01*T12 + P02*T22);
    const S vSv10 = -(P01*T00 + P11*T10 + P12*T20);
    const S vSv11 = -(P01*T01 + P11*T11 + P12*T21);
    const S vSv12 = -(P01*T02 + P11*T12 + P12*T22);
    const S vSv20 = -(P02*T00 + P12*T10 + P22*T20);
    const S vSv21 = -(P02*T01 + P12*T11 + P22*T21);
    const S vSv22 = -(P02*T02 + P12*T12 + P22*T22);

    // v_covars: diagonal = v_Sv[i][i] * r[i]^2
    //           off-diag = (v_Sv[i][j] + v_Sv[j][i]) * r[i]*r[j]
    v_covars[idx * 6 + 0] = vSv00 * r0 * r0;
    v_covars[idx * 6 + 1] = (vSv01 + vSv10) * r0 * r1;
    v_covars[idx * 6 + 2] = (vSv02 + vSv20) * r0 * r2;
    v_covars[idx * 6 + 3] = vSv11 * r1 * r1;
    v_covars[idx * 6 + 4] = (vSv12 + vSv21) * r1 * r2;
    v_covars[idx * 6 + 5] = vSv22 * r2 * r2;
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

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
    cudaStream_t stream
) {
    if (N == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (N + threads - 1) / threads;

    world_to_volume_mapping_bwd_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, ranges, centers,
        dim2, dim1, dim0, eps3d,
        v_means3d_v, v_conics,
        v_means, v_covars, N
    );
}

// Explicit template instantiations
template void launch_world_to_volume_mapping_bwd<float>(
    const float*, const float*, const float*, const float*,
    uint32_t, uint32_t, uint32_t, float,
    const float*, const float*,
    float*, float*, uint32_t, cudaStream_t
);
template void launch_world_to_volume_mapping_bwd<double>(
    const double*, const double*, const double*, const double*,
    uint32_t, uint32_t, uint32_t, double,
    const double*, const double*,
    double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
