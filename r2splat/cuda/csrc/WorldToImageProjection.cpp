/**
 * @file WorldToImageProjection.cpp
 * @brief C++ Tensor Wrapper for World-to-Image Projection Module
 *
 * Provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 *
 * Input shapes:
 *   means:     [N, 3]    -- world-space Gaussian centers (shared across cameras)
 *   covars:    [N, 6]    -- world-space covariance upper-tri (shared across cameras)
 *   viewmats:  [C, 4, 4] -- camera extrinsic matrices
 *   Ks:        [C, 3, 3] -- camera intrinsic matrices
 *
 * Output shapes:
 *   means2d:   [C, N, 2]  -- projected 2D centers
 *   cov2d_3x3: [C, N, 6]  -- projected 3x3 covariance (upper-tri)
 *   radii:     [C, N, 2]  -- axis-aligned tile radii (rx, ry)
 *
 * Following r2splat's pattern for tensor handling.
 *
 * Author: Dy164
 * Date: 2026-02-19
 */

#include "WorldToImageProjection.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_image_projection_fwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    float eps_blur,
    float radius_clip
) {
    R2SPLAT_DEVICE_GUARD(means);
    R2SPLAT_CHECK_INPUT(means);
    R2SPLAT_CHECK_INPUT(covars);
    R2SPLAT_CHECK_INPUT(viewmats);
    R2SPLAT_CHECK_INPUT(Ks);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3], got shape ",
                means.sizes());
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6], got shape ",
                covars.sizes());
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
                "viewmats must be 3D [C, 4, 4], got shape ",
                viewmats.sizes());
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(1) == 3 && Ks.size(2) == 3,
                "Ks must be 3D [C, 3, 3], got shape ",
                Ks.sizes());
    TORCH_CHECK(radius_clip > 0.0f,
                "radius_clip must be positive, got ", radius_clip);

    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars dim 0 (", covars.size(0),
                ") doesn't match N=", N);
    TORCH_CHECK(Ks.size(0) == C,
                "Ks batch dim (", Ks.size(0),
                ") doesn't match C=", C);

    // Allocate outputs
    torch::Tensor means2d = torch::empty(
        {C, N, 2}, means.options());
    torch::Tensor cov2d_3x3 = torch::empty(
        {C, N, 6}, means.options());
    torch::Tensor radii = torch::zeros(
        {C, N, 2}, means.options().dtype(torch::kInt32));

    if (total == 0) {
        return std::make_tuple(means2d, cov2d_3x3, radii);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_image_projection_fwd",
        [&]() {
            launch_world_to_image_projection_fwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                dim1,
                dim0,
                static_cast<scalar_t>(eps_blur),
                static_cast<scalar_t>(radius_clip),
                means2d.data_ptr<scalar_t>(),
                cov2d_3x3.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(means2d, cov2d_3x3, radii);
}

// ============================================================================
// First-Order Backward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
world_to_image_projection_bwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& viewmats,    // [C, 4, 4]
    const torch::Tensor& Ks,          // [C, 3, 3]
    uint32_t dim1,
    uint32_t dim0,
    const torch::Tensor& v_means2d,   // [C, N, 2]
    const torch::Tensor& v_cov2d_3x3  // [C, N, 6]
) {
    R2SPLAT_DEVICE_GUARD(means);
    R2SPLAT_CHECK_INPUT(means);
    R2SPLAT_CHECK_INPUT(covars);
    R2SPLAT_CHECK_INPUT(viewmats);
    R2SPLAT_CHECK_INPUT(Ks);
    R2SPLAT_CHECK_INPUT(v_means2d);
    R2SPLAT_CHECK_INPUT(v_cov2d_3x3);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3]");
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6]");
    TORCH_CHECK(viewmats.dim() == 3 && viewmats.size(1) == 4 && viewmats.size(2) == 4,
                "viewmats must be 3D [C, 4, 4]");
    TORCH_CHECK(Ks.dim() == 3 && Ks.size(1) == 3 && Ks.size(2) == 3,
                "Ks must be 3D [C, 3, 3]");
    TORCH_CHECK(v_means2d.dim() == 3 && v_means2d.size(2) == 2,
                "v_means2d must be 3D [C, N, 2]");
    TORCH_CHECK(v_cov2d_3x3.dim() == 3 && v_cov2d_3x3.size(2) == 6,
                "v_cov2d_3x3 must be 3D [C, N, 6]");

    const int64_t N = means.size(0);
    const int64_t C = viewmats.size(0);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars dim 0 mismatch");
    TORCH_CHECK(v_means2d.size(0) == C && v_means2d.size(1) == N,
                "v_means2d shape mismatch");
    TORCH_CHECK(v_cov2d_3x3.size(0) == C && v_cov2d_3x3.size(1) == N,
                "v_cov2d_3x3 shape mismatch");

    // Allocate outputs: accumulated over C cameras via atomicAdd in kernel
    torch::Tensor v_means = torch::empty(
        {N, 3}, means.options());
    torch::Tensor v_covars = torch::empty(
        {N, 6}, means.options());

    if (total == 0) {
        return std::make_tuple(v_means, v_covars);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_image_projection_bwd",
        [&]() {
            launch_world_to_image_projection_bwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                viewmats.data_ptr<scalar_t>(),
                Ks.data_ptr<scalar_t>(),
                dim1,
                dim0,
                v_means2d.data_ptr<scalar_t>(),
                v_cov2d_3x3.data_ptr<scalar_t>(),
                v_means.data_ptr<scalar_t>(),
                v_covars.data_ptr<scalar_t>(),
                static_cast<uint32_t>(C),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(v_means, v_covars);
}

} // namespace r2splat
