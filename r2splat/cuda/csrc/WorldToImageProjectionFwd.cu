/**
 * @file WorldToImageProjectionFwd.cu
 * @brief Forward CUDA kernel: world-to-image projection
 *
 * Fuses world-to-camera coordinate transform with perspective projection
 * to compute 2D means and projected 3x3 covariance from world-space inputs.
 *
 * =========================================================================
 * Mathematical Derivation
 * =========================================================================
 *
 * Given:
 *   p     = (px, py, pz)   -- world-space Gaussian center
 *   Sigma = 3x3 symmetric  -- world-space covariance (upper-tri: s0..s5)
 *   R     = 3x3 rotation   -- camera rotation (from viewmat)
 *   t     = 3x1 translation -- camera translation (from viewmat)
 *   fx, fy, cx, cy          -- camera intrinsics
 *
 * Step 1: Camera coordinate transform
 *   p_c = R * p + t = (x, y, z)  in camera frame
 *
 * Step 2: Perspective projection for means2d
 *   rz = 1/z
 *   means2d = (fx * x * rz + cx,  fy * y * rz + cy)
 *
 * Step 3: FOV clipping (hard clamp, following gsplat reference)
 *   tx = z * clamp(x/z, -lim_neg_x, lim_pos_x)
 *   ty = z * clamp(y/z, -lim_neg_y, lim_pos_y)
 *   where lim_pos_x = (W - cx)/fx + 0.3 * tan(fov_x/2), etc.
 *
 * Step 4: Build Jacobian J (3x3)
 *   l = sqrt(tx^2 + ty^2 + z^2)
 *   J = [[fx/z,   0,      -fx*tx/z^2],
 *        [0,      fy/z,   -fy*ty/z^2],
 *        [tx/l,   ty/l,    z/l      ]]
 *
 * Step 5: Camera covariance
 *   Sigma_c = R * Sigma * R^T
 *
 * Step 6: Projected covariance
 *   cov2d_3x3 = J * Sigma_c * J^T
 *   Store as upper-triangular (a, b, c, d, e, f)
 *
 * Each thread processes one (camera, Gaussian) pair.
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#include "WorldToImageProjection.h"
#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Helper: Unpack upper-triangular 6 elements to symmetric 3x3 matrix
// ============================================================================
template <typename S>
__device__ __forceinline__ void unpack_sym3x3(
    const S* __restrict__ tri6,
    S& m00, S& m01, S& m02,
    S& m11, S& m12, S& m22
) {
    m00 = tri6[0]; m01 = tri6[1]; m02 = tri6[2];
    m11 = tri6[3]; m12 = tri6[4]; m22 = tri6[5];
}

// ============================================================================
// Forward Kernel
// ============================================================================

template <typename S>
__global__ void world_to_image_projection_fwd_kernel(
    const S* __restrict__ means,       // [N, 3]
    const S* __restrict__ covars,      // [N, 6]
    const S* __restrict__ viewmats,    // [C, 4, 4] row-major
    const S* __restrict__ Ks,          // [C, 3, 3] row-major
    uint32_t dim1,
    uint32_t dim0,
    S eps_blur, // note that eps_blur is equal to eps2d in ConicsMu
    S radius_clip,
    S* __restrict__ means2d,           // [C*N, 2]
    S* __restrict__ cov2d_3x3,        // [C*N, 6]
    int32_t* __restrict__ radii,       // [C*N, 2]
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
    // viewmat is [C, 4, 4] row-major: first 3 rows are [R | t]
    const S* vm = viewmats + cam_id * 16;
    // R (row-major in viewmat):
    //   vm[0] vm[1] vm[2]
    //   vm[4] vm[5] vm[6]
    //   vm[8] vm[9] vm[10]
    const S R00 = vm[0], R01 = vm[1], R02 = vm[2];
    const S R10 = vm[4], R11 = vm[5], R12 = vm[6];
    const S R20 = vm[8], R21 = vm[9], R22 = vm[10];
    const S t0 = vm[3], t1 = vm[7], t2 = vm[11];

    // Intrinsics: Ks is [C, 3, 3] row-major
    const S* K = Ks + cam_id * 9;
    const S fx = K[0], fy = K[4];
    const S cx = K[2], cy = K[5];

    // ================================================================
    // Read world-space Gaussian (shared across cameras, indexed by gs_id)
    // ================================================================
    const S* p = means + gs_id * 3;
    const S px = p[0], py = p[1], pz = p[2];

    // ================================================================
    // Step 1: Camera transform  p_c = R * p + t
    // ================================================================
    const S x = R00 * px + R01 * py + R02 * pz + t0;
    const S y = R10 * px + R11 * py + R12 * pz + t1;
    const S z = R20 * px + R21 * py + R22 * pz + t2;

    // Skip if behind camera
    if (z <= static_cast<S>(0.01)) {
        means2d[idx * 2 + 0] = static_cast<S>(0);
        means2d[idx * 2 + 1] = static_cast<S>(0);
        for (int i = 0; i < 6; ++i) cov2d_3x3[idx * 6 + i] = static_cast<S>(0);
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // ================================================================
    // Step 2: Perspective projection for means2d
    // ================================================================
    const S rz = static_cast<S>(1) / z;
    const S rz2 = rz * rz;
    means2d[idx * 2 + 0] = fx * x * rz + cx;
    means2d[idx * 2 + 1] = fy * y * rz + cy;

    // ================================================================
    // Step 3: FOV clipping (hard clamp)
    // ================================================================
    const S W = static_cast<S>(dim1);
    const S H = static_cast<S>(dim0);
    const S tan_fovx = static_cast<S>(0.5) * W / fx;
    const S tan_fovy = static_cast<S>(0.5) * H / fy;
    const S lim_x_pos = (W - cx) / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_x_neg = cx / fx + static_cast<S>(0.3) * tan_fovx;
    const S lim_y_pos = (H - cy) / fy + static_cast<S>(0.3) * tan_fovy;
    const S lim_y_neg = cy / fy + static_cast<S>(0.3) * tan_fovy;

    const S tx = z * min(lim_x_pos, max(-lim_x_neg, x * rz));
    const S ty = z * min(lim_y_pos, max(-lim_y_neg, y * rz));

    // ================================================================
    // Step 4: Build Jacobian J (3x3)
    // ================================================================
    const S l = sqrt(tx * tx + ty * ty + z * z);
    const S rl = static_cast<S>(1) / l;

    // J in row-major:
    //   J00 = fx*rz,  J01 = 0,      J02 = -fx*tx*rz2
    //   J10 = 0,      J11 = fy*rz,  J12 = -fy*ty*rz2
    //   J20 = tx*rl,  J21 = ty*rl,  J22 = z*rl
    const S J00 = fx * rz;
    const S J02 = -fx * tx * rz2;
    const S J11 = fy * rz;
    const S J12 = -fy * ty * rz2;
    const S J20 = tx * rl;
    const S J21 = ty * rl;
    const S J22 = z * rl;
    // J01 = J10 = 0

    // ================================================================
    // Step 5: Camera covariance  Sigma_c = R * Sigma * R^T
    // ================================================================
    // Read world covariance (upper-tri, shared across cameras, indexed by gs_id)
    S s00, s01, s02, s11, s12, s22;
    unpack_sym3x3(covars + gs_id * 6, s00, s01, s02, s11, s12, s22);

    // M = Sigma * R^T  (R^T columns = R rows, since R row-major)
    // M[i][j] = sum_k Sigma[i][k] * R[j][k]
    // We use the symmetric property of Sigma.
    const S M00 = s00*R00 + s01*R01 + s02*R02;
    const S M01 = s00*R10 + s01*R11 + s02*R12;
    const S M02 = s00*R20 + s01*R21 + s02*R22;
    const S M10 = s01*R00 + s11*R01 + s12*R02;
    const S M11 = s01*R10 + s11*R11 + s12*R12;
    const S M12 = s01*R20 + s11*R21 + s12*R22;
    const S M20 = s02*R00 + s12*R01 + s22*R02;
    const S M21 = s02*R10 + s12*R11 + s22*R12;
    const S M22 = s02*R20 + s12*R21 + s22*R22;

    // Sigma_c = R * M  (R is row-major: R[i][j] = R_{ij})
    // Sigma_c[i][j] = sum_k R[i][k] * M[k][j]
    const S Sc00 = R00*M00 + R01*M10 + R02*M20;
    const S Sc01 = R00*M01 + R01*M11 + R02*M21;
    const S Sc02 = R00*M02 + R01*M12 + R02*M22;
    const S Sc11 = R10*M01 + R11*M11 + R12*M21;
    const S Sc12 = R10*M02 + R11*M12 + R12*M22;
    const S Sc22 = R20*M02 + R21*M12 + R22*M22;

    // ================================================================
    // Step 6: Projected covariance  C2 = J * Sigma_c * J^T
    // ================================================================
    // N = Sigma_c * J^T  (J^T column j = J row j)
    // N[i][j] = sum_k Sigma_c[i][k] * J[j][k]
    // Using symmetry of Sigma_c:
    const S N00 = Sc00*J00 +              Sc02*J02;  // J01=0
    const S N01 = Sc00*static_cast<S>(0) + Sc01*J11 + Sc02*J12;
    const S N02 = Sc00*J20 + Sc01*J21 + Sc02*J22;
    const S N10 = Sc01*J00 +              Sc12*J02;
    const S N11 = Sc01*static_cast<S>(0) + Sc11*J11 + Sc12*J12;
    const S N12 = Sc01*J20 + Sc11*J21 + Sc12*J22;
    const S N20 = Sc02*J00 +              Sc22*J02;
    const S N21 = Sc02*static_cast<S>(0) + Sc12*J11 + Sc22*J12;
    const S N22 = Sc02*J20 + Sc12*J21 + Sc22*J22;

    // C2[i][j] = sum_k J[i][k] * N[k][j]
    const S C00 = J00*N00 +              J02*N20;
    const S C01 = J00*N01 +              J02*N21;
    const S C02 = J00*N02 +              J02*N22;
    const S C11 =              J11*N11 + J12*N21;
    const S C12 =              J11*N12 + J12*N22;
    const S C22 = J20*N02 + J21*N12 + J22*N22;

    // ================================================================
    // Write output: cov2d_3x3 as upper-triangular 6 elements
    // ================================================================
    S* out_cov = cov2d_3x3 + idx * 6;
    out_cov[0] = C00;  // a = Sigma_tilde[0][0]
    out_cov[1] = C01;  // b = Sigma_tilde[0][1]
    out_cov[2] = C02;  // c = Sigma_tilde[0][2]
    out_cov[3] = C11;  // d = Sigma_tilde[1][1]
    out_cov[4] = C12;  // e = Sigma_tilde[1][2]
    out_cov[5] = C22;  // f = Sigma_tilde[2][2]

    // ================================================================
    // Compute axis-aligned rectangular radii from the 2D covariance with blur
    // (non-differentiable). For q <= radius_clip^2, the exact AABB half-widths
    // are radius_clip * sqrt(Sigma_xx) and radius_clip * sqrt(Sigma_yy).
    // Following fully_fused_projection_fwd.cu from 3DGS2CT reference
    // ================================================================
    const S a_blur = C00 + eps_blur;
    const S d_blur = C11 + eps_blur;
    const S b_off = C01;
    const S det_blur = a_blur * d_blur - b_off * b_off;
    if (det_blur <= static_cast<S>(0)) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    const S radius_x = ceil(radius_clip * sqrt(max(static_cast<S>(0), a_blur)));
    const S radius_y = ceil(radius_clip * sqrt(max(static_cast<S>(0), d_blur)));

    if (!isfinite(radius_x) || !isfinite(radius_y) ||
        radius_x <= static_cast<S>(0) || radius_y <= static_cast<S>(0)) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    const S mean_x = means2d[idx * 2];
    const S mean_y = means2d[idx * 2 + 1];
    if (mean_x + radius_x <= static_cast<S>(0) ||
        mean_x - radius_x >= static_cast<S>(dim1) ||
        mean_y + radius_y <= static_cast<S>(0) ||
        mean_y - radius_y >= static_cast<S>(dim0)) {
        radii[idx * 2 + 0] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    radii[idx * 2 + 0] = static_cast<int32_t>(radius_x);
    radii[idx * 2 + 1] = static_cast<int32_t>(radius_y);
}

// ============================================================================
// Launch Function Implementation
// ============================================================================

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
    cudaStream_t stream
) {
    const uint32_t total = C * N;
    if (total == 0) return;
    const int threads = R2SPLAT_N_THREADS;
    const int blocks = (total + threads - 1) / threads;

    world_to_image_projection_fwd_kernel<S><<<blocks, threads, 0, stream>>>(
        means, covars, viewmats, Ks,
        dim1, dim0, eps_blur, radius_clip,
        means2d, cov2d_3x3, radii,
        C, N
    );
}

// Explicit template instantiations
template void launch_world_to_image_projection_fwd<float>(
    const float*, const float*, const float*, const float*,
    uint32_t, uint32_t, float, float,
    float*, float*, int32_t*,
    uint32_t, uint32_t, cudaStream_t
);
template void launch_world_to_image_projection_fwd<double>(
    const double*, const double*, const double*, const double*,
    uint32_t, uint32_t, double, double,
    double*, double*, int32_t*,
    uint32_t, uint32_t, cudaStream_t
);

} // namespace r2splat
