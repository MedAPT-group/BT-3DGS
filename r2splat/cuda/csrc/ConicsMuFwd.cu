/**
 * @file ConicsMuFwd.cu
 * @brief Forward CUDA kernel: compute conics and mu from projected 3x3 covariance
 *        with blur (eps * I) applied before inversion.
 * 
 * Given the projected 3x3 covariance matrix tilde{Sigma} stored as 6 upper-triangular
 * elements (a, b, c, d, e, f), computes:
 *   1. conics = (hat{Sigma} + eps*I_2)^{-1}, stored as (u11, u12, u22)
 *   2. mu = sqrt(2*pi * det(tilde{Sigma} + eps*I_3) / det(hat{Sigma} + eps*I_2))
 * 
 * The blur ensures numerical stability (prevents singular inversions) and provides
 * anti-aliasing for sub-voxel Gaussians (Mip-Splatting style).
 * 
 * Note: cov3d_inv = (tilde{Sigma} + eps*I_3)^{-1} is NOT computed here to avoid
 * unnecessary global memory writes. It is recomputed locally in the backward pass
 * where it is actually needed for efficient gradient computation.
 * 
 * Each thread processes one Gaussian (element of the C*N batch).
 * 
 * Author: Dy164
 * Date: 2026-02-10
 */

#include "ConicsMu.h"
#include "../include/common.h"
#include <cuda_runtime.h>
#include <math_constants.h>

namespace r2splat {

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename S>
__global__ void conics_mu_fwd_kernel(
    const S* __restrict__ cov2d_3x3,   // [total, 6]  (a, b, c, d, e, f)
    const S eps,                       // blur epsilon
    S* __restrict__ conics,            // [total, 3]  (u11, u12, u22)
    S* __restrict__ mus,               // [total]
    uint32_t total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    // Read the 6 independent elements of tilde{Sigma}
    const S* cov = cov2d_3x3 + idx * 6;
    const S a = cov[0];  // tilde{Sigma}_{00}
    const S b = cov[1];  // tilde{Sigma}_{01}
    const S c = cov[2];  // tilde{Sigma}_{02}
    const S d = cov[3];  // tilde{Sigma}_{11}
    const S e = cov[4];  // tilde{Sigma}_{12}
    const S f = cov[5];  // tilde{Sigma}_{22}

    // Apply blur: add eps to diagonal elements
    const S ab = a + eps;  // (a + eps)
    const S db = d + eps;  // (d + eps)
    const S fb = f + eps;  // (f + eps)
    // Off-diagonal elements remain: b, c, e

    // ----------------------------------------------------------------
    // Blurred 2x2 sub-block: hat{Sigma}_b = [[ab, b], [b, db]]
    // ----------------------------------------------------------------
    const S D2b = ab * db - b * b;

    // ----------------------------------------------------------------
    // Blurred full 3x3: tilde{Sigma}_b = [[ab, b, c], [b, db, e], [c, e, fb]]
    // ----------------------------------------------------------------
    const S D3b = ab * db * fb - ab * e * e - b * b * fb
                + static_cast<S>(2) * b * c * e - c * c * db;

    // Validity check
    const S validity_eps = static_cast<S>(1e-9);
    if (D2b <= validity_eps) {
        // Degenerate 2D covariance, output zeros
        conics[idx * 3 + 0] = static_cast<S>(0);
        conics[idx * 3 + 1] = static_cast<S>(0);
        conics[idx * 3 + 2] = static_cast<S>(0);
        mus[idx] = static_cast<S>(0);
        return;
    }

    // ================================================================
    // Compute conics = hat{Sigma}_b^{-1}
    // ================================================================
    const S inv_D2b = static_cast<S>(1) / D2b;
    conics[idx * 3 + 0] =  db * inv_D2b;   // u11
    conics[idx * 3 + 1] = -b  * inv_D2b;   // u12
    conics[idx * 3 + 2] =  ab * inv_D2b;   // u22

    // ================================================================
    // Compute mu = sqrt(2*pi * D3b / D2b)
    // ================================================================
    const S ratio = D3b * inv_D2b;
    if (ratio > static_cast<S>(0)) {
        mus[idx] = sqrt(static_cast<S>(2.0 * CUDART_PI) * ratio);
    } else {
        mus[idx] = static_cast<S>(0);
    }
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

template <typename S>
void launch_conics_mu_fwd(
    const S* __restrict__ cov2d_3x3,
    S eps,
    S* __restrict__ conics,
    S* __restrict__ mus,
    uint32_t total,
    cudaStream_t stream
) {
    if (total == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    conics_mu_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        cov2d_3x3, eps, conics, mus, total
    );
}

// Explicit template instantiations
template void launch_conics_mu_fwd<float>(
    const float*, float, float*, float*, uint32_t, cudaStream_t
);
template void launch_conics_mu_fwd<double>(
    const double*, double, double*, double*, uint32_t, cudaStream_t
);

} // namespace r2splat
