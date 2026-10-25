/**
 * @file ParameterTransform.cpp
 * @brief C++ Tensor Wrapper for Parameter Transform (quat, scale_ori) -> covar
 *
 * Provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 *
 * Input shapes:
 *   quats:     [N, 4]  -- raw quaternions (w, x, y, z), normalized in-kernel
 *   scale_ori: [N, 3]  -- raw scale parameters (before sigmoid activation)
 *
 * Output shapes:
 *   covars:    [N, 6]  -- world-space covariance (upper-triangular)
 *
 * Following r2splat's pattern for tensor handling.
 *
 * Author: Dy164
 * Date: 2026-02-20
 */

#include "ParameterTransform.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_fwd_tensor(
    const torch::Tensor& quats,       // [N, 4]
    const torch::Tensor& scale_ori,   // [N, 3]
    double scale_min,
    double scale_max
) {
    R2SPLAT_DEVICE_GUARD(quats);
    R2SPLAT_CHECK_INPUT(quats);
    R2SPLAT_CHECK_INPUT(scale_ori);

    // Validate dimensions
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4,
                "quats must be 2D [N, 4], got shape ", quats.sizes());
    TORCH_CHECK(scale_ori.dim() == 2 && scale_ori.size(1) == 3,
                "scale_ori must be 2D [N, 3], got shape ", scale_ori.sizes());

    const int64_t N = quats.size(0);
    TORCH_CHECK(scale_ori.size(0) == N,
                "scale_ori batch dim (", scale_ori.size(0),
                ") doesn't match quats N=", N);

    // Allocate outputs
    torch::Tensor covars = torch::empty({N, 6}, quats.options());
    torch::Tensor max_scales = torch::empty({N}, quats.options());

    if (N == 0) {
        return std::make_tuple(covars, max_scales);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        quats.scalar_type(),
        "parameter_transform_fwd",
        [&]() {
            launch_parameter_transform_fwd<scalar_t>(
                quats.data_ptr<scalar_t>(),
                scale_ori.data_ptr<scalar_t>(),
                static_cast<scalar_t>(scale_min),
                static_cast<scalar_t>(scale_max),
                covars.data_ptr<scalar_t>(),
                max_scales.data_ptr<scalar_t>(),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(covars, max_scales);
}

// ============================================================================
// First-Order Backward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
parameter_transform_bwd_tensor(
    const torch::Tensor& quats,       // [N, 4]
    const torch::Tensor& scale_ori,   // [N, 3]
    double scale_min,
    double scale_max,
    const torch::Tensor& v_covars     // [N, 6]
) {
    R2SPLAT_DEVICE_GUARD(quats);
    R2SPLAT_CHECK_INPUT(quats);
    R2SPLAT_CHECK_INPUT(scale_ori);
    R2SPLAT_CHECK_INPUT(v_covars);

    // Validate dimensions
    TORCH_CHECK(quats.dim() == 2 && quats.size(1) == 4,
                "quats must be 2D [N, 4]");
    TORCH_CHECK(scale_ori.dim() == 2 && scale_ori.size(1) == 3,
                "scale_ori must be 2D [N, 3]");
    TORCH_CHECK(v_covars.dim() == 2 && v_covars.size(1) == 6,
                "v_covars must be 2D [N, 6]");

    const int64_t N = quats.size(0);
    TORCH_CHECK(scale_ori.size(0) == N && v_covars.size(0) == N,
                "Batch dimension mismatch");

    // Allocate outputs
    torch::Tensor v_quats = torch::empty({N, 4}, quats.options());
    torch::Tensor v_scale_ori = torch::empty({N, 3}, quats.options());

    if (N == 0) {
        return std::make_tuple(v_quats, v_scale_ori);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        quats.scalar_type(),
        "parameter_transform_bwd",
        [&]() {
            launch_parameter_transform_bwd<scalar_t>(
                quats.data_ptr<scalar_t>(),
                scale_ori.data_ptr<scalar_t>(),
                static_cast<scalar_t>(scale_min),
                static_cast<scalar_t>(scale_max),
                v_covars.data_ptr<scalar_t>(),
                v_quats.data_ptr<scalar_t>(),
                v_scale_ori.data_ptr<scalar_t>(),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(v_quats, v_scale_ori);
}

} // namespace r2splat
