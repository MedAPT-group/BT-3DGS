/**
 * @file ConicsMuBwd.cu
 * @brief Backward CUDA kernel: gradients of conics & mu w.r.t. projected 3x3 covariance
 *        using efficient conics-based formulation (with blur support).
 * 
 * Given downstream gradients dL/d(conics) = (v_u11, v_u12, v_u22) and dL/d(mu) = v_mu,
 * computes dL/d(cov2d_3x3) = (dL/da, dL/db, dL/dc, dL/dd, dL/de, dL/df) via chain rule.
 * 
 * This backward pass recomputes cov3d_inv = (tilde{Sigma}_b)^{-1} locally in each
 * thread from the original cov2d_3x3 and eps, avoiding the need to store it between
 * forward and backward. The arithmetic cost is minimal (cofactor expansion for a
 * symmetric 3x3 matrix), while the memory saving is significant (one fewer [C*N, 6]
 * tensor to allocate and transfer through global memory).
 * 
 * --- Efficient Formulation ---
 * 
 * Let P = (hat{Sigma}_b)^{-1} = conics = (u11, u12, u22)
 * Let Q = (tilde{Sigma}_b)^{-1} = recomputed locally = (q00, q01, q02, q11, q12, q22)
 * 
 * Part 1: Gradient from conics
 *   d(P_ij)/d(Sigma_kl) = -P_ik * P_lj
 *   For independent elements of the symmetric matrix:
 *     d[u11,u12,u22]/d[a,b,d] = -[[u11^2,     2*u11*u12,          u12^2     ],
 *                                  [u11*u12,   u11*u22 + u12^2,    u12*u22   ],
 *                                  [u12^2,     2*u12*u22,          u22^2     ]]
 *   Conics do not depend on c, e, f.
 * 
 * Part 2: Gradient from mu
 *   d(mu)/d[a,b,c,d,e,f] = (mu/2) * ([q00, 2*q01, 2*q02, q11, 2*q12, q22]
 *                                   - [u11, 2*u12, 0,     u22, 0,     0   ])
 *   This uses the identity: d(log det A)/d(A) = A^{-1}
 *   with appropriate handling of symmetric matrix independent elements
 *   (off-diagonal elements get factor 2).
 * 
 * Author: Dy164
 * Date: 2026-02-10
 */

#include "ConicsMu.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Backward Kernel (First-Order Gradients, Efficient Formulation)
// ============================================================================

template <typename S>
__global__ void conics_mu_bwd_kernel(
    const S* __restrict__ cov2d_3x3,   // [total, 6]  (a, b, c, d, e, f)
    const S eps,                       // blur epsilon
    const S* __restrict__ conics,      // [total, 3]  (u11, u12, u22)
    const S* __restrict__ mus,         // [total]
    const S* __restrict__ v_conics,    // [total, 3]  (v_u11, v_u12, v_u22)
    const S* __restrict__ v_mus,       // [total]
    S* __restrict__ v_cov2d_3x3,      // [total, 6]
    uint32_t total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // ---- Read forward outputs (conics already blur-aware) ----
    const S u11 = conics[idx * 3 + 0];
    const S u12 = conics[idx * 3 + 1];
    const S u22 = conics[idx * 3 + 2];
    const S mu  = mus[idx];

    // ---- Read downstream gradients ----
    const S v_u11 = v_conics[idx * 3 + 0];
    const S v_u12 = v_conics[idx * 3 + 1];
    const S v_u22 = v_conics[idx * 3 + 2];
    const S v_mu  = v_mus[idx];

    // Output pointer
    S* v_cov = v_cov2d_3x3 + idx * 6;

    // Handle degenerate case (conics are all zero from forward)
    if (u11 == static_cast<S>(0) && u12 == static_cast<S>(0) && u22 == static_cast<S>(0)) {
        v_cov[0] = static_cast<S>(0);
        v_cov[1] = static_cast<S>(0);
        v_cov[2] = static_cast<S>(0);
        v_cov[3] = static_cast<S>(0);
        v_cov[4] = static_cast<S>(0);
        v_cov[5] = static_cast<S>(0);
        return;
    }

    // ==================================================================
    // Part 1: Gradient contribution from conics (u11, u12, u22)
    // ==================================================================
    // Using dP_ij / dSigma_kl = -P_ik * P_lj
    // For symmetric matrix independent elements (off-diagonal b has factor 2):
    //
    // dL/da = v_u11 * (-u11^2)    + v_u12 * (-u11*u12)    + v_u22 * (-u12^2)
    // dL/db = v_u11 * (-2*u11*u12) + v_u12 * (-(u11*u22+u12^2)) + v_u22 * (-2*u12*u22)
    // dL/dc = 0 (conics independent of c)
    // dL/dd = v_u11 * (-u12^2)    + v_u12 * (-u12*u22)    + v_u22 * (-u22^2)
    // dL/de = 0 (conics independent of e)
    // dL/df = 0 (conics independent of f)

    S grad_a = -(v_u11 * u11 * u11 + v_u12 * u11 * u12 + v_u22 * u12 * u12);
    S grad_b = -(v_u11 * static_cast<S>(2) * u11 * u12
              + v_u12 * (u11 * u22 + u12 * u12)
              + v_u22 * static_cast<S>(2) * u12 * u22);
    S grad_c = static_cast<S>(0);
    S grad_d = -(v_u11 * u12 * u12 + v_u12 * u12 * u22 + v_u22 * u22 * u22);
    S grad_e = static_cast<S>(0);
    S grad_f = static_cast<S>(0);

    // ==================================================================
    // Part 2: Gradient contribution from mu
    // ==================================================================
    // d(mu)/d[a,b,c,d,e,f] = (mu/2) * ([q00, 2*q01, 2*q02, q11, 2*q12, q22]
    //                                 - [u11, 2*u12, 0,     u22, 0,     0   ])
    //
    // Derivation:
    //   mu = sqrt(2*pi * D3b / D2b)
    //   d(mu)/d(Sigma) = (mu/2) * (d(log D3b)/d(Sigma) - d(log D2b)/d(Sigma))
    //   d(log det A)/d(A) = A^{-1}  (for symmetric positive definite A)
    //   => d(log D3b)/d(Sigma) = tilde{Sigma}_b^{-1} = Q  (recomputed below)
    //      d(log D2b)/d(Sigma) = pad(hat{Sigma}_b^{-1}) = pad(conics)
    //   For independent elements of symmetric matrix:
    //     diagonal: factor 1, off-diagonal: factor 2

    if (mu > static_cast<S>(0)) {
        // ---- Recompute cov3d_inv locally from cov2d_3x3 + eps ----
        const S* cov = cov2d_3x3 + idx * 6;
        const S ab = cov[0] + eps;   // a + eps
        const S b  = cov[1];         // b
        const S c  = cov[2];         // c
        const S db = cov[3] + eps;   // d + eps
        const S e  = cov[4];         // e
        const S fb = cov[5] + eps;   // f + eps

        const S D3b = ab * db * fb - ab * e * e - b * b * fb
                    + static_cast<S>(2) * b * c * e - c * c * db;

        const S validity_eps = static_cast<S>(1e-9);
        if (D3b > validity_eps) {
            const S inv_D3b = static_cast<S>(1) / D3b;

            // Cofactor expansion for symmetric 3x3 inverse
            const S q00 = (db * fb - e * e)  * inv_D3b;
            const S q01 = (c * e  - b * fb)  * inv_D3b;
            const S q02 = (b * e  - c * db)  * inv_D3b;
            const S q11 = (ab * fb - c * c)  * inv_D3b;
            const S q12 = (b * c  - ab * e)  * inv_D3b;
            const S q22 = (ab * db - b * b)  * inv_D3b;  // = D2b / D3b

            const S half_mu_v = static_cast<S>(0.5) * mu * v_mu;

            grad_a += half_mu_v * (q00 - u11);
            grad_b += half_mu_v * (static_cast<S>(2) * q01 - static_cast<S>(2) * u12);
            grad_c += half_mu_v * (static_cast<S>(2) * q02);
            grad_d += half_mu_v * (q11 - u22);
            grad_e += half_mu_v * (static_cast<S>(2) * q12);
            grad_f += half_mu_v * q22;
        }
        // If D3b <= validity_eps, mu contribution is zero (degenerate 3x3)
    }

    // ---- Write output ----
    v_cov[0] = grad_a;
    v_cov[1] = grad_b;
    v_cov[2] = grad_c;
    v_cov[3] = grad_d;
    v_cov[4] = grad_e;
    v_cov[5] = grad_f;
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_conics_mu_bwd(
    const S* __restrict__ cov2d_3x3,
    S eps,
    const S* __restrict__ conics,
    const S* __restrict__ mus,
    const S* __restrict__ v_conics,
    const S* __restrict__ v_mus,
    S* __restrict__ v_cov2d_3x3,
    uint32_t total,
    cudaStream_t stream
) {
    if (total == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    conics_mu_bwd_kernel<S><<<blocks, threads, 0, stream>>>(
        cov2d_3x3, eps, conics, mus, v_conics, v_mus, v_cov2d_3x3, total
    );
}

// Explicit template instantiations
template void launch_conics_mu_bwd<float>(
    const float*, float, const float*, const float*,
    const float*, const float*, float*, uint32_t, cudaStream_t
);
template void launch_conics_mu_bwd<double>(
    const double*, double, const double*, const double*,
    const double*, const double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
