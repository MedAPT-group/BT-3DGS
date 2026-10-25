/**
 * @file ConicsMu.cpp
 * @brief C++ Tensor Wrapper for Conics & Mu Computation (with Blur)
 * 
 * Provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 
 * Input shape: [C, N, 6] — C cameras, N Gaussians, 6 upper-triangular elements
 * Output conics shape: [C, N, 3] — (hat{Sigma}_b)^{-1}
 * Output mu shape: [C, N] — sqrt(2*pi * det(tilde{Sigma}_b) / det(hat{Sigma}_b))
 * 
 * The forward pass applies blur (eps * I) to the covariance before inversion.
 * The backward pass recomputes cov3d_inv = (tilde{Sigma}_b)^{-1} locally from
 * the original cov2d_3x3 and eps, avoiding extra memory allocation in forward.
 * 
 * Following r2splat's pattern for tensor handling.
 * 
 * Author: Dy164
 * Date: 2026-02-10
 */

#include "ConicsMu.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
conics_mu_fwd_tensor(
    const torch::Tensor& cov2d_3x3,   // [C, N, 6]
    double eps                          // blur epsilon
) {
    R2SPLAT_DEVICE_GUARD(cov2d_3x3);
    R2SPLAT_CHECK_INPUT(cov2d_3x3);

    // Validate dimensions
    TORCH_CHECK(cov2d_3x3.dim() == 3,
                "cov2d_3x3 must be 3D [C, N, 6], got dim=", cov2d_3x3.dim());
    TORCH_CHECK(cov2d_3x3.size(2) == 6,
                "cov2d_3x3 last dim must be 6, got ", cov2d_3x3.size(2));

    const int64_t C = cov2d_3x3.size(0);
    const int64_t N = cov2d_3x3.size(1);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Allocate outputs
    torch::Tensor conics = torch::empty(
        {C, N, 3}, cov2d_3x3.options());
    torch::Tensor mus = torch::empty(
        {C, N}, cov2d_3x3.options());

    if (total == 0) {
        return std::make_tuple(conics, mus);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        cov2d_3x3.scalar_type(),
        "conics_mu_fwd",
        [&]() {
            launch_conics_mu_fwd<scalar_t>(
                cov2d_3x3.data_ptr<scalar_t>(),
                static_cast<scalar_t>(eps),
                conics.data_ptr<scalar_t>(),
                mus.data_ptr<scalar_t>(),
                total,
                stream
            );
        }
    );

    return std::make_tuple(conics, mus);
}

// ============================================================================
// First-Order Backward Pass Tensor Interface
// ============================================================================

torch::Tensor
conics_mu_bwd_tensor(
    const torch::Tensor& cov2d_3x3,   // [C, N, 6]  (forward input, needed for recomputation)
    double eps,                         // blur epsilon (same value as forward)
    const torch::Tensor& conics,       // [C, N, 3]
    const torch::Tensor& mus,          // [C, N]
    const torch::Tensor& v_conics,     // [C, N, 3]
    const torch::Tensor& v_mus         // [C, N]
) {
    R2SPLAT_DEVICE_GUARD(cov2d_3x3);
    R2SPLAT_CHECK_INPUT(cov2d_3x3);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(mus);
    R2SPLAT_CHECK_INPUT(v_conics);
    R2SPLAT_CHECK_INPUT(v_mus);

    // Validate dimensions
    TORCH_CHECK(cov2d_3x3.dim() == 3,
                "cov2d_3x3 must be 3D [C, N, 6]");
    TORCH_CHECK(conics.dim() == 3,
                "conics must be 3D [C, N, 3]");
    TORCH_CHECK(mus.dim() == 2,
                "mus must be 2D [C, N]");
    TORCH_CHECK(v_conics.dim() == 3,
                "v_conics must be 3D [C, N, 3]");
    TORCH_CHECK(v_mus.dim() == 2,
                "v_mus must be 2D [C, N]");

    const int64_t C = cov2d_3x3.size(0);
    const int64_t N = cov2d_3x3.size(1);
    const uint32_t total = static_cast<uint32_t>(C * N);

    // Validate shape consistency
    TORCH_CHECK(cov2d_3x3.size(2) == 6,
                "cov2d_3x3 last dim must be 6");
    TORCH_CHECK(conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3,
                "conics shape must be [C, N, 3]");
    TORCH_CHECK(mus.size(0) == C && mus.size(1) == N,
                "mus shape must be [C, N]");
    TORCH_CHECK(v_conics.size(0) == C && v_conics.size(1) == N && v_conics.size(2) == 3,
                "v_conics shape must be [C, N, 3]");
    TORCH_CHECK(v_mus.size(0) == C && v_mus.size(1) == N,
                "v_mus shape must be [C, N]");

    // Allocate output gradient
    torch::Tensor v_cov2d_3x3 = torch::empty(
        {C, N, 6}, cov2d_3x3.options());

    if (total == 0) {
        return v_cov2d_3x3;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        cov2d_3x3.scalar_type(),
        "conics_mu_bwd",
        [&]() {
            launch_conics_mu_bwd<scalar_t>(
                cov2d_3x3.data_ptr<scalar_t>(),
                static_cast<scalar_t>(eps),
                conics.data_ptr<scalar_t>(),
                mus.data_ptr<scalar_t>(),
                v_conics.data_ptr<scalar_t>(),
                v_mus.data_ptr<scalar_t>(),
                v_cov2d_3x3.data_ptr<scalar_t>(),
                total,
                stream
            );
        }
    );

    return v_cov2d_3x3;
}

} // namespace r2splat
