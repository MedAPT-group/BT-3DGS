/**
 * @file WorldToImageProjectionBwd.cu
 * @brief First-order backward CUDA kernel for world-to-image projection
 *
 * Given downstream gradients dL/d(means2d) and dL/d(cov2d_3x3),
 * computes upstream gradients dL/d(means3d) and dL/d(covars) via chain rule.
 *
 * =========================================================================
 * Mathematical Derivation (First-Order Backward)
 * =========================================================================
 *
 * The forward pass computes:
 *   means2d = f(p)       where p = means3d
 *   cov2d_3x3 = g(p, Sigma) where Sigma = covars
 *
 * The chain rule gives:
 *   v_p = v_means2d^T @ (d means2d / d p)
 *       + v_cov2d_3x3^T @ (d cov2d_3x3 / d p)
 *   v_Sigma = v_cov2d_3x3^T @ (d cov2d_3x3 / d Sigma)
 *
 * --- Gradient of means2d w.r.t. means3d ---
 *
 * means2d = (fx * x * rz + cx,  fy * y * rz + cy)
 * where (x, y, z) = R * p + t,  rz = 1/z
 *
 * d(means2d)/d(p_c) = [[fx*rz,  0,       -fx*x*rz^2],
 *                       [0,      fy*rz,   -fy*y*rz^2]]
 * (This is the 2x3 sub-matrix F of J)
 *
 * d(means2d)/d(p) = F * R
 * => v_p += R^T * F^T * v_means2d
 *
 * --- Gradient of cov2d_3x3 w.r.t. covars ---
 *
 * cov2d_3x3 = J * R * Sigma * R^T * J^T
 * Let P = J * R, then cov2d_3x3 = P * Sigma * P^T
 *
 * Using the identity for D = P * X * P^T with grad G = v_cov2d_3x3:
 *   v_Sigma = P^T * G * P
 *   (for symmetric matrices, we need to handle off-diagonal properly)
 *
 * For the 6-element independent parameterization:
 *   v_sigma_k = sum_{i<=j} v_C_ij * d(C_ij)/d(sigma_k)
 *   where off-diagonal elements of Sigma appear twice.
 *
 * --- Gradient of cov2d_3x3 w.r.t. means3d (through J) ---
 *
 * J depends on p_c (through tx, ty, z, l).
 * This generates gradient contributions to v_p from v_cov2d_3x3.
 * Using v_J = v_cov2d_3x3 * J * Sigma_c^T + v_cov2d_3x3^T * J * Sigma_c
 * then propagate v_J through J -> p_c -> p.
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#include "WorldToImageProjection.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// First-Order Backward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_image_projection_bwd_kernel(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4]
    const S* __restrict__ Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const S* __restrict__ v_means2d,   // [C*N, 2]
    const S* __restrict__ v_cov2d_3x3, // [C*N, 6]
    S* __restrict__ v_means_out,       // [N, 3] (atomicAdd accumulation)
    S* __restrict__ v_covars_out,      // [N, 6] (atomicAdd accumulation)
    uint32_t C,
    uint32_t N
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = C * N;
    if (idx >= total) return;

    const uint32_t cam_id = idx / N;
    const uint32_t gs_id = idx % N;

    // ================================================================
    // Read camera parameters
    // ================================================================
    const S* vm = viewmats + cam_id * 16;
    const S R00 = vm[0], R01 = vm[1], R02 = vm[2];
    const S R10 = vm[4], R11 = vm[5], R12 = vm[6];
    const S R20 = vm[8], R21 = vm[9], R22 = vm[10];
    const S t0 = vm[3], t1 = vm[7], t2 = vm[11];

    const S* K = Ks + cam_id * 9;
    const S fx = K[0], fy = K[4];
    const S cx = K[2], cy = K[5];

    // ================================================================
    // Read world-space Gaussian (shared across cameras, indexed by gs_id)
    // ================================================================
    const S* p = means + gs_id * 3;
    const S px = p[0], py = p[1], pz = p[2];

    // Read world covariance (shared across cameras, indexed by gs_id)
    const S* cov_ptr = covars + gs_id * 6;
    const S s00 = cov_ptr[0], s01 = cov_ptr[1], s02 = cov_ptr[2];
    const S s11 = cov_ptr[3], s12 = cov_ptr[4], s22 = cov_ptr[5];

    // ================================================================
    // Recompute forward quantities
    // ================================================================
    const S x = R00 * px + R01 * py + R02 * pz + t0;
    const S y = R10 * px + R11 * py + R12 * pz + t1;
    const S z = R20 * px + R21 * py + R22 * pz + t2;

    // Use local variables for per-thread gradient computation
    S v_p_local[3];
    S v_s_local[6];

    // Handle behind-camera case
    if (z <= static_cast<S>(0.01)) {
        // No contribution to gradients from this (camera, gaussian) pair
        return;
    }

    const S rz = static_cast<S>(1) / z;
    const S rz2 = rz * rz;
    const S rz3 = rz2 * rz;

    // FOV clipping
    const S W = static_cast<S>(dim1);
    const S H = static_cast<S>(dim0);
    const S tan_fovx = static_cast<S>(0.5) * W / fx;
    const S tan_fovy = static_cast<S>(0.5) * H / fy;
    const S lim_x_pos = (W - cx) / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_x_neg = cx / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_y_pos = (H - cy) / fy + static_cast<S>(0.3) * tan_fovy;
    const S lim_y_neg = cy / fy + static_cast<S>(0.3) * tan_fovy;

    const S xrz = x * rz;
    const S yrz = y * rz;
    const S tx = z * min(lim_x_pos, max(-lim_x_neg, xrz));
    const S ty = z * min(lim_y_pos, max(-lim_y_neg, yrz));

    const bool x_clipped = (xrz > lim_x_pos) || (xrz < -lim_x_neg);
    const bool y_clipped = (yrz > lim_y_pos) || (yrz < -lim_y_neg);

    // Build J
    const S l = sqrt(tx * tx + ty * ty + z * z);
    const S rl = static_cast<S>(1) / l;
    const S rl2 = rl * rl;

    const S J00 = fx * rz;
    const S J02 = -fx * tx * rz2;
    const S J11 = fy * rz;
    const S J12 = -fy * ty * rz2;
    const S J20 = tx * rl;
    const S J21 = ty * rl;
    const S J22 = z * rl;

    // ================================================================
    // Camera covariance: Sigma_c = R * Sigma * R^T
    // ================================================================
    // M = Sigma * R^T
    const S M00 = s00*R00 + s01*R01 + s02*R02;
    const S M01 = s00*R10 + s01*R11 + s02*R12;
    const S M02 = s00*R20 + s01*R21 + s02*R22;
    const S M10 = s01*R00 + s11*R01 + s12*R02;
    const S M11 = s01*R10 + s11*R11 + s12*R12;
    const S M12 = s01*R20 + s11*R21 + s12*R22;
    const S M20 = s02*R00 + s12*R01 + s22*R02;
    const S M21 = s02*R10 + s12*R11 + s22*R12;
    const S M22 = s02*R20 + s12*R21 + s22*R22;

    // Sigma_c = R * M
    const S Sc00 = R00*M00 + R01*M10 + R02*M20;
    const S Sc01 = R00*M01 + R01*M11 + R02*M21;
    const S Sc02 = R00*M02 + R01*M12 + R02*M22;
    const S Sc11 = R10*M01 + R11*M11 + R12*M21;
    const S Sc12 = R10*M02 + R11*M12 + R12*M22;
    const S Sc22 = R20*M02 + R21*M12 + R22*M22;

    // ================================================================
    // Read downstream gradients
    // ================================================================
    const S vm2_x = v_means2d[idx * 2 + 0];
    const S vm2_y = v_means2d[idx * 2 + 1];

    const S* vc = v_cov2d_3x3 + idx * 6;
    // v_cov2d_3x3 = (va, vb, vc, vd, ve, vf) for upper-tri parameterization.
    //
    // The packed output is: cov2d_packed[k] = C2[i_k][j_k]  for i_k <= j_k.
    // Since each off-diagonal element C2[i][j] is stored ONCE (not twice),
    // the gradient from autograd is an UPPER-TRIANGULAR matrix G_upper:
    //   G_upper[i][j] = v_k  for i <= j,  G_upper[i][j] = 0  for i > j.
    //
    // For Part B (v_covars): v_Sigma_c = J^T * G_upper * J  (NOT symmetric!)
    // For Part C (v_J):      v_J = (G_upper + G_upper^T) * J * Sigma_c
    //
    // We store both G_upper (for Part B) and G+G^T (for Part C) separately.
    const S va = vc[0], vb = vc[1], vc_val = vc[2];
    const S vd = vc[3], ve = vc[4], vf = vc[5];

    // G_upper: upper-triangular gradient matrix (for Part B: v_Sigma_c)
    const S vC00 = va,               vC01 = vb,     vC02 = vc_val;
    const S vC10 = static_cast<S>(0), vC11 = vd,     vC12 = ve;
    const S vC20 = static_cast<S>(0), vC21 = static_cast<S>(0), vC22 = vf;

    // ================================================================
    // Part A: Gradient from means2d → means3d
    // ================================================================
    // v_p_c from means2d:
    //   v_pc.x = fx * rz * vm2_x
    //   v_pc.y = fy * rz * vm2_y
    //   v_pc.z = -(fx*x*rz2*vm2_x + fy*y*rz2*vm2_y)
    S v_pc_x = fx * rz * vm2_x;
    S v_pc_y = fy * rz * vm2_y;
    S v_pc_z = -(fx * x * rz2 * vm2_x + fy * y * rz2 * vm2_y);

    // ================================================================
    // Part B: Gradient from cov2d_3x3 → covars (through Sigma)
    // ================================================================
    // cov2d_3x3_packed[k] = C2[i_k][j_k] where C2 = J * Sigma_c * J^T
    // v_Sigma_c = J^T * G_upper * J  (G_upper is upper-triangular, NOT symmetric)
    //
    // First: T = G_upper * J  (3x3 * 3x3 = 3x3)
    //   T[i][j] = sum_k vC[i][k] * J[k][j]
    // J has sparse structure: J01 = J10 = 0
    // G_upper has vC10 = vC20 = vC21 = 0
    const S T00 = vC00*J00 +              vC02*J20;
    const S T01 =              vC01*J11 + vC02*J21;
    const S T02 = vC00*J02 +  vC01*J12 + vC02*J22;
    const S T10 = vC10*J00 +              vC12*J20;
    const S T11 =              vC11*J11 + vC12*J21;
    const S T12 = vC10*J02 +  vC11*J12 + vC12*J22;
    const S T20 = vC20*J00 +              vC22*J20;
    const S T21 =              vC21*J11 + vC22*J21;
    const S T22 = vC20*J02 +  vC21*J12 + vC22*J22;

    // v_Sigma_c[i][j] = sum_k J[k][i] * T[k][j]  (J^T * T)
    // Since J sparse: J^T row 0 = (J00, 0, J20), etc.
    const S vSc00 = J00*T00 +              J20*T20;
    const S vSc01 = J00*T01 +              J20*T21;
    const S vSc02 = J00*T02 +              J20*T22;
    const S vSc10 =              J11*T10 + J21*T20;
    const S vSc11 =              J11*T11 + J21*T21;
    const S vSc12 =              J11*T12 + J21*T22;
    const S vSc20 = J02*T00 + J12*T10 + J22*T20;
    const S vSc21 = J02*T01 + J12*T11 + J22*T21;
    const S vSc22 = J02*T02 + J12*T12 + J22*T22;

    // v_Sigma_c = J^T * G_upper * J is NOT symmetric (since G_upper is not).
    // We need the FULL 3x3 result to propagate through R.
    // (No symmetrization here — we'll extract vS[p][q]+vS[q][p] at the end.)

    // v_Sigma (world) = R^T * v_Sigma_c * R
    // v_Sigma_c is NOT symmetric, so v_Sigma is also NOT symmetric.
    // For the upper-tri parameterization of Sigma:
    //   diagonal:  v_covars[m] = vS[p][p]
    //   off-diag:  v_covars[m] = vS[p][q] + vS[q][p]  (for p < q)

    // Q = v_Sigma_c * R  (3x3 times 3x3)
    // Using v_Sigma_c = (vSc_{ij})
    const S Q00 = vSc00*R00 + vSc01*R10 + vSc02*R20;
    const S Q01 = vSc00*R01 + vSc01*R11 + vSc02*R21;
    const S Q02 = vSc00*R02 + vSc01*R12 + vSc02*R22;
    const S Q10 = vSc10*R00 + vSc11*R10 + vSc12*R20;
    const S Q11 = vSc10*R01 + vSc11*R11 + vSc12*R21;
    const S Q12 = vSc10*R02 + vSc11*R12 + vSc12*R22;
    const S Q20 = vSc20*R00 + vSc21*R10 + vSc22*R20;
    const S Q21 = vSc20*R01 + vSc21*R11 + vSc22*R21;
    const S Q22 = vSc20*R02 + vSc21*R12 + vSc22*R22;

    // v_Sigma_full = R^T * Q  (full 3x3, NOT symmetric)
    const S vS00 = R00*Q00 + R10*Q10 + R20*Q20;
    const S vS01 = R00*Q01 + R10*Q11 + R20*Q21;
    const S vS02 = R00*Q02 + R10*Q12 + R20*Q22;
    const S vS10 = R01*Q00 + R11*Q10 + R21*Q20;
    const S vS11 = R01*Q01 + R11*Q11 + R21*Q21;
    const S vS12 = R01*Q02 + R11*Q12 + R21*Q22;
    const S vS20 = R02*Q00 + R12*Q10 + R22*Q20;
    const S vS21 = R02*Q01 + R12*Q11 + R22*Q21;
    const S vS22 = R02*Q02 + R12*Q12 + R22*Q22;

    // Write v_covars (6 independent elements of the upper-tri parameterization)
    // Since v_Sigma is NOT symmetric (because G_upper is upper-triangular):
    //   diagonal:  v_covars[m] = vS[p][p]
    //   off-diag:  v_covars[m] = vS[p][q] + vS[q][p]  (for p < q)
    v_s_local[0] = vS00;                // d/d(s00) - diagonal
    v_s_local[1] = vS01 + vS10;         // d/d(s01) - off-diagonal
    v_s_local[2] = vS02 + vS20;         // d/d(s02) - off-diagonal
    v_s_local[3] = vS11;                // d/d(s11) - diagonal
    v_s_local[4] = vS12 + vS21;         // d/d(s12) - off-diagonal
    v_s_local[5] = vS22;                // d/d(s22) - diagonal

    // ================================================================
    // Part C: Gradient from cov2d_3x3 → means3d (through J)
    // ================================================================
    // cov2d_3x3 = J * Sigma_c * J^T
    // v_J = v_C2d * J * Sigma_c^T + v_C2d^T * J * Sigma_c
    //      (since Sigma_c is symmetric and v_C2d is symmetric):
    // v_J = 2 * v_C2d * J * Sigma_c  ... NO, this isn't right either.
    //
    // For D = A * X * A^T with grad G:
    //   v_A = G * A * X^T + G^T * A * X = (G + G^T) * A * X  (when X symmetric)
    //       = 2 * G * A * X  (when G is also symmetric)
    // Wait: G * A * X^T + G^T * A * X. With G = G^T and X = X^T:
    //   = G * A * X + G * A * X = 2 * G * A * X.
    // So v_J = 2 * v_C2d * J * Sigma_c

    // Actually, let's be more careful. The standard formula is:
    // For D = A * X * A^T, with dL/dD = G:
    //   dL/dA = G * A * X^T + G^T * A * X
    // When G is symmetric (G = G^T) and X is symmetric (X = X^T):
    //   dL/dA = G * A * X + G * A * X = 2 * G * A * X
    // Hmm, that doesn't seem right either since G * A * X^T = G * A * X when X symmetric.
    // So dL/dA = 2 * G * A * X.

    // Let me verify with a different approach. D_ij = sum_k,l A_ik X_kl A_jl
    // dD_ij/dA_mn = delta_im * sum_l X_nl A_jl + delta_jm * sum_k A_ik X_kn
    //             = delta_im * (X A^T)_nj + delta_jm * (A X)_in
    // dL/dA_mn = sum_ij G_ij * dD_ij/dA_mn
    //          = sum_j G_mj (X A^T)_nj + sum_i G_im (AX)_in
    //          = (G * A * X^T)_mn + (G^T * A * X)_mn
    // With G symmetric and X symmetric:
    //          = (G * A * X)_mn + (G * A * X)_mn = 2 (G * A * X)_mn

    // So v_J = 2 * v_C2d * J * Sigma_c (when both are symmetric, which they are)

    // But wait - v_C2d is the gradient matrix from the 6-element parameterization.
    // We constructed it as the FULL symmetric matrix. So yes, this is correct.
    // Actually, I need to be more careful. Our v_C2d (vC00..vC22) is constructed as:
    //   vC[i][j] = vC[j][i] = the gradient of the upper-tri element.
    // This is indeed the gradient of the full matrix w.r.t. loss, treating the
    // matrix as a symmetric matrix with 6 independent parameters.
    // The formula v_J = G * A * X^T + G^T * A * X = 2 G A X applies when G is
    // the gradient of the FULL matrix (not the independent parameterization).
    // But our G IS the full matrix gradient. So this is correct.

    // Wait, actually our G is NOT quite the full matrix gradient. When we have
    // L = f(D) where D is symmetric and parameterized by 6 elements:
    //   dL/dD_full[i][j] = dL/d(sigma_k) * 1  for i<=j, (i,j)->k
    //   dL/dD_full[j][i] = dL/d(sigma_k) * 1  for i<j
    // So the full matrix gradient has:
    //   G_full[i][j] = dL/d(sigma_k) for all (i,j) pairs
    // This means G_full[0][1] = dL/d(s01) = vb, G_full[1][0] = vb as well.
    // This is exactly what we constructed. So we're good.

    // Hmm, but actually there's a subtlety. The standard formula assumes
    // D_ij are all independent. When D is symmetric with 6 independent params:
    //   dL/d(A) = ?
    // Let's derive directly:
    //   sigma_k = D[i][j] = sum_a,b A_ia S_ab A_jb  (where S = Sigma_c)
    //   For k corresponding to (i,j) with i<=j:
    //     dL/d(A_mn) = sum_k v_k * d(sigma_k)/dA_mn
    //   For diagonal k=(i,i):
    //     d(sigma_k)/dA_mn = 2 delta_im sum_b S_nb A_ib  (since D_ii = sum_a,b A_ia S_ab A_ib)
    //     Actually: d(D_ii)/dA_mn = delta_im * (SA^T)_ni + delta_im * (AS)_in
    //            = 2 * delta_im * sum_b S_nb A_ib  (when S symmetric)
    //   For off-diagonal k=(i,j) with i<j:
    //     sigma_k = D[i][j] = sum_a,b A_ia S_ab A_jb
    //     d(sigma_k)/dA_mn = delta_im * (SA^T)_nj + delta_jm * (AS)_in ... hmm
    //
    // The correct gradient formula for D = A * X * A^T with dL/dD = G is:
    //   dL/dA = G * A * X^T + G^T * A * X = (G + G^T) * A * X  (when X sym)
    //
    // IMPORTANT: G here is the gradient from the 6-element upper-tri
    // parameterization. Since cov2d_3x3 packs only upper-tri elements
    // from the full 3x3 matrix, autograd produces an UPPER-TRIANGULAR
    // gradient G (lower triangle = 0), NOT a symmetric one.
    //
    // G_upper:                         G + G^T:
    //   [[va,     vb,     vc_val],       [[2*va,   vb,     vc_val],
    //    [0,      vd,     ve    ],        [vb,     2*vd,   ve    ],
    //    [0,      0,      vf    ]]        [vc_val, ve,     2*vf  ]]
    //
    // Note: 2*G_sym (what was previously used) would double the off-diags,
    // giving [[2*va, 2*vb, 2*vc], [2*vb, 2*vd, 2*ve], [2*vc, 2*ve, 2*vf]]
    // which is WRONG. The off-diagonal elements should appear only once.
    //
    // Build (G + G^T):
    const S GGt00 = static_cast<S>(2) * va;
    const S GGt01 = vb;
    const S GGt02 = vc_val;
    const S GGt10 = vb;
    const S GGt11 = static_cast<S>(2) * vd;
    const S GGt12 = ve;
    const S GGt20 = vc_val;
    const S GGt21 = ve;
    const S GGt22 = static_cast<S>(2) * vf;

    // Compute U = J * Sigma_c (3x3 times 3x3, using Sigma_c symmetric)
    const S U00 = J00*Sc00 +              J02*Sc02;
    const S U01 = J00*Sc01 +              J02*Sc12;
    const S U02 = J00*Sc02 +              J02*Sc22;
    const S U10 =              J11*Sc01 + J12*Sc02;
    const S U11 =              J11*Sc11 + J12*Sc12;
    const S U12 =              J11*Sc12 + J12*Sc22;
    const S U20 = J20*Sc00 + J21*Sc01 + J22*Sc02;
    const S U21 = J20*Sc01 + J21*Sc11 + J22*Sc12;
    const S U22 = J20*Sc02 + J21*Sc12 + J22*Sc22;

    // v_J = (G + G^T) * U  (3x3 * 3x3)
    const S vJ00 = GGt00*U00 + GGt01*U10 + GGt02*U20;
    const S vJ01 = GGt00*U01 + GGt01*U11 + GGt02*U21;
    const S vJ02 = GGt00*U02 + GGt01*U12 + GGt02*U22;
    const S vJ10 = GGt10*U00 + GGt11*U10 + GGt12*U20;
    const S vJ11 = GGt10*U01 + GGt11*U11 + GGt12*U21;
    const S vJ12 = GGt10*U02 + GGt11*U12 + GGt12*U22;
    const S vJ20 = GGt20*U00 + GGt21*U10 + GGt22*U20;
    const S vJ21 = GGt20*U01 + GGt21*U11 + GGt22*U21;
    const S vJ22 = GGt20*U02 + GGt21*U12 + GGt22*U22;

    // ================================================================
    // Propagate v_J → v_p_c
    // ================================================================
    // J depends on p_c = (x, y, z) through tx, ty, l, rz.
    //
    // J elements:
    //   J00 = fx * rz           -> dJ00/dz = -fx * rz^2
    //   J02 = -fx * tx * rz^2   -> depends on tx and z
    //   J11 = fy * rz           -> dJ11/dz = -fy * rz^2
    //   J12 = -fy * ty * rz^2   -> depends on ty and z
    //   J20 = tx / l            -> depends on tx and l
    //   J21 = ty / l            -> depends on ty and l
    //   J22 = z / l             -> depends on z and l
    //
    // tx = z * clamp(x/z, ...)
    //   If not clipped: tx = x, dtx/dx = 1, dtx/dz = 0
    //   If clipped: tx = z * lim, dtx/dx = 0, dtx/dz = lim
    // Similarly for ty.

    // Gradient of l w.r.t. tx, ty, z:
    //   dl/dtx = tx/l, dl/dty = ty/l, dl/dz_direct = z/l

    // Propagate v_J through J → p_c (following gsplat reference pattern):

    // From J[0][0] = fx * rz:  contributes to v_pc.z
    v_pc_z += -fx * rz2 * vJ00;

    // From J[1][1] = fy * rz:  contributes to v_pc.z
    v_pc_z += -fy * rz2 * vJ11;

    // From J[2][2] = z / l:
    v_pc_z += rl * vJ22;

    // From J[0][2] = -fx * tx * rz^2:
    //   dJ02/dx = -fx * (dtx/dx) * rz^2 = -fx * rz^2  (if not clipped)
    //   dJ02/dz = -fx * (dtx/dz * rz^2 + tx * (-2*rz^3))
    //           = -fx * (0 * rz^2 - 2*tx*rz^3) = 2*fx*tx*rz^3  (if not clipped)
    //   If clipped: dJ02/dx = 0
    //               dJ02/dz = -fx * (lim * rz^2 - 2*tx*rz^3)
    if (!x_clipped) {
        v_pc_x += -fx * rz2 * vJ02;
    }
    v_pc_z += static_cast<S>(2) * fx * tx * rz3 * vJ02;

    // From J[1][2] = -fy * ty * rz^2:
    if (!y_clipped) {
        v_pc_y += -fy * rz2 * vJ12;
    }
    v_pc_z += static_cast<S>(2) * fy * ty * rz3 * vJ12;

    // From J[2][0] = tx / l:
    //   dJ20/dx = (dtx/dx)/l - tx*(dl/dtx)*(dtx/dx)/l^2
    //           = 1/l - tx*(tx/l)/l^2 = 1/l - tx^2/l^3  (if not clipped)
    //           = (l^2 - tx^2)/l^3
    //   dJ20/dz:  through tx (if clipped: dtx/dz = lim) and l
    if (!x_clipped) {
        v_pc_x += rl * vJ20;  // From dtx/dx=1 in tx/l
    }

    // From J[2][1] = ty / l:
    if (!y_clipped) {
        v_pc_y += rl * vJ21;
    }

    // Calculate v_l (gradient through l = sqrt(tx^2 + ty^2 + z^2))
    // l appears in J20, J21, J22 as 1/l
    S v_l = -tx * rl2 * vJ20 - ty * rl2 * vJ21 - z * rl2 * vJ22;

    // Propagate v_l through l → tx, ty, z
    // dl/dtx = tx/l, dl/dty = ty/l, dl/dz = z/l
    if (!x_clipped) {
        v_pc_x += v_l * tx * rl;  // dl/dtx * dtx/dx (=1)
    }
    if (!y_clipped) {
        v_pc_y += v_l * ty * rl;  // dl/dty * dty/dy (=1)
    }
    v_pc_z += v_l * z * rl;  // dl/dz

    // If clipped, tx is a function of z only (tx = z * lim)
    // The contributions from dtx/dz * ... need to be added to v_pc_z:
    if (x_clipped) {
        // tx = z * lim_x, dtx/dz = lim_x = tx/z
        // From J02 = -fx * tx * rz^2:
        //   Additional term: -fx * (tx/z) * rz^2 * vJ02 (already partially included)
        //   Actually the 2*fx*tx*rz3 term above handles the z part through rz.
        //   But we need the dJ20/dz through tx:
        //   J20 = tx/l, dtx/dz = tx/z
        //   dJ20/dz_via_tx = (tx/z)/l - tx*(tx/l * tx/z)/l^2
        //                  = tx/(z*l) * (1 - tx^2/l^2)
        // And dl/dtx * dtx/dz = (tx/l) * (tx/z) = tx^2 / (l*z)
        v_pc_z += v_l * tx * rl * (tx / z);
        // dJ20/dtx * dtx/dz:
        //   dJ20/dtx = 1/l - tx^2/l^3 = (l^2-tx^2)/l^3
        //   but we already have vJ20, the rl contribution needs the dtx/dz factor
        // Actually: simpler to just do:
        //   vJ20 * d(tx/l)/dz_via_tx = vJ20 * (dtx/dz * l - tx * dl/dtx * dtx/dz) / l^2
        //   = vJ20 * ((tx/z) * l - tx * (tx/l) * (tx/z)) / l^2
        //   = vJ20 * (tx/(z*l)) * (1 - tx^2/l^2)
        // But we've already separated out v_l and the direct rl terms.
        // Let me follow gsplat's simpler approach:
        // When x is clipped, dJ20/dx = 0 (already handled), and
        // the z-contribution from tx:
        // dJ02/dz_via_tx_clipped: already part of the rz3 term
        // dJ20/dz_via_tx_clipped = (dtx/dz)/l = (tx/z)/l
        v_pc_z += (tx / z) * rl * vJ20;
    }
    if (y_clipped) {
        v_pc_z += v_l * ty * rl * (ty / z);
        v_pc_z += (ty / z) * rl * vJ21;
    }

    // Handle the clipped case for J02 and J12:
    // When x is clipped: dJ02/dz includes dtx/dz = tx/z term
    // -fx * dtx/dz * rz^2 = -fx * (tx/z) * rz^2
    if (x_clipped) {
        v_pc_z += -fx * (tx / z) * rz2 * vJ02;
    }
    if (y_clipped) {
        v_pc_z += -fy * (ty / z) * rz2 * vJ12;
    }

    // ================================================================
    // Transform v_p_c to v_p (world space): v_p = R^T * v_p_c
    // Accumulate into [N, 3] output using atomicAdd (shared across cameras)
    // ================================================================
    v_p_local[0] = R00 * v_pc_x + R10 * v_pc_y + R20 * v_pc_z;
    v_p_local[1] = R01 * v_pc_x + R11 * v_pc_y + R21 * v_pc_z;
    v_p_local[2] = R02 * v_pc_x + R12 * v_pc_y + R22 * v_pc_z;

    // Atomically accumulate gradients across cameras into [N, ...] outputs
    for (int i = 0; i < 3; ++i) {
        atomicAdd(v_means_out + gs_id * 3 + i, v_p_local[i]);
    }
    for (int i = 0; i < 6; ++i) {
        atomicAdd(v_covars_out + gs_id * 6 + i, v_s_local[i]);
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

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
    cudaStream_t stream
) {
    const uint32_t total = C * N;
    if (total == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    // Zero-initialize output tensors (atomicAdd accumulation requires zeroed memory)
    cudaMemsetAsync(v_means, 0, N * 3 * sizeof(S), stream);
    cudaMemsetAsync(v_covars, 0, N * 6 * sizeof(S), stream);

    world_to_image_projection_bwd_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, viewmats, Ks,
        dim1, dim0,
        v_means2d, v_cov2d_3x3,
        v_means, v_covars,
        C, N
    );
}

// Explicit template instantiations
template void launch_world_to_image_projection_bwd<float>(
    const float*, const float*, const float*, const float*,
    uint32_t, uint32_t,
    const float*, const float*,
    float*, float*,
    uint32_t, uint32_t, cudaStream_t
);
template void launch_world_to_image_projection_bwd<double>(
    const double*, const double*, const double*, const double*,
    uint32_t, uint32_t,
    const double*, const double*,
    double*, double*,
    uint32_t, uint32_t, cudaStream_t
);

} // namespace r2splat
