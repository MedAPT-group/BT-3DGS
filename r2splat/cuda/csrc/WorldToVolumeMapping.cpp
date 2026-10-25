/**
 * @file WorldToVolumeMapping.cpp
 * @brief C++ Tensor Wrapper for World-to-Volume Mapping Module
 *
 * Provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 *
 * Input shapes (note: NO batch dimension C, unlike WorldToImageProjection):
 *   means:      [N, 3]  -- world-space Gaussian centers
 *   covars:     [N, 6]  -- world-space covariance (upper-tri)
 *   max_scales: [N]     -- max activated scale per Gaussian
 *   ranges:     [3]     -- physical extent per axis
 *   centers:    [3]     -- physical center per axis
 *
 * Output shapes:
 *   means3d_v:  [N, 3]  -- voxel-space centers
 *   conics:     [N, 6]  -- inverse covariance (upper-tri)
 *   radii:      [N, 3]  -- integer radii per axis (int32)
 *
 * Author: Dy164
 * Date: 2026-02-21
 */

#include "WorldToVolumeMapping.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
world_to_volume_mapping_fwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& max_scales,  // [N]
    const torch::Tensor& ranges,      // [3]
    const torch::Tensor& centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    double radius_clip
) {
    R2SPLAT_DEVICE_GUARD(means);
    R2SPLAT_CHECK_INPUT(means);
    R2SPLAT_CHECK_INPUT(covars);
    R2SPLAT_CHECK_INPUT(max_scales);
    R2SPLAT_CHECK_INPUT(ranges);
    R2SPLAT_CHECK_INPUT(centers);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3], got shape ", means.sizes());
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6], got shape ", covars.sizes());
    TORCH_CHECK(max_scales.dim() == 1,
                "max_scales must be 1D [N], got dim ", max_scales.dim());
    TORCH_CHECK(ranges.dim() == 1 && ranges.size(0) == 3,
                "ranges must be 1D [3], got shape ", ranges.sizes());
    TORCH_CHECK(centers.dim() == 1 && centers.size(0) == 3,
                "centers must be 1D [3], got shape ", centers.sizes());

    const int64_t N = means.size(0);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N,
                "covars batch dim (", covars.size(0),
                ") doesn't match N=", N);
    TORCH_CHECK(max_scales.size(0) == N,
                "max_scales batch dim (", max_scales.size(0),
                ") doesn't match N=", N);

    // Allocate outputs
    torch::Tensor means3d_v = torch::empty({N, 3}, means.options());
    torch::Tensor conics = torch::empty({N, 6}, means.options());
    torch::Tensor radii = torch::empty({N, 3}, means.options().dtype(torch::kInt32));

    if (N == 0) {
        return std::make_tuple(means3d_v, conics, radii);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_volume_mapping_fwd",
        [&]() {
            launch_world_to_volume_mapping_fwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                max_scales.data_ptr<scalar_t>(),
                ranges.data_ptr<scalar_t>(),
                centers.data_ptr<scalar_t>(),
                dim2, dim1, dim0,
                static_cast<scalar_t>(eps3d),
                static_cast<scalar_t>(radius_clip),
                means3d_v.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(means3d_v, conics, radii);
}

// ============================================================================
// First-Order Backward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor>
world_to_volume_mapping_bwd_tensor(
    const torch::Tensor& means,       // [N, 3]
    const torch::Tensor& covars,      // [N, 6]
    const torch::Tensor& ranges,      // [3]
    const torch::Tensor& centers,     // [3]
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    double eps3d,
    const torch::Tensor& v_means3d_v, // [N, 3]
    const torch::Tensor& v_conics     // [N, 6]
) {
    R2SPLAT_DEVICE_GUARD(means);
    R2SPLAT_CHECK_INPUT(means);
    R2SPLAT_CHECK_INPUT(covars);
    R2SPLAT_CHECK_INPUT(ranges);
    R2SPLAT_CHECK_INPUT(centers);
    R2SPLAT_CHECK_INPUT(v_means3d_v);
    R2SPLAT_CHECK_INPUT(v_conics);

    // Validate dimensions
    TORCH_CHECK(means.dim() == 2 && means.size(1) == 3,
                "means must be 2D [N, 3]");
    TORCH_CHECK(covars.dim() == 2 && covars.size(1) == 6,
                "covars must be 2D [N, 6]");
    TORCH_CHECK(ranges.dim() == 1 && ranges.size(0) == 3,
                "ranges must be 1D [3]");
    TORCH_CHECK(centers.dim() == 1 && centers.size(0) == 3,
                "centers must be 1D [3]");
    TORCH_CHECK(v_means3d_v.dim() == 2 && v_means3d_v.size(1) == 3,
                "v_means3d_v must be 2D [N, 3]");
    TORCH_CHECK(v_conics.dim() == 2 && v_conics.size(1) == 6,
                "v_conics must be 2D [N, 6]");

    const int64_t N = means.size(0);

    // Validate shape consistency
    TORCH_CHECK(covars.size(0) == N, "covars shape mismatch");
    TORCH_CHECK(v_means3d_v.size(0) == N, "v_means3d_v shape mismatch");
    TORCH_CHECK(v_conics.size(0) == N, "v_conics shape mismatch");

    // Allocate outputs
    torch::Tensor v_means = torch::empty({N, 3}, means.options());
    torch::Tensor v_covars = torch::empty({N, 6}, means.options());

    if (N == 0) {
        return std::make_tuple(v_means, v_covars);
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(
        means.scalar_type(),
        "world_to_volume_mapping_bwd",
        [&]() {
            launch_world_to_volume_mapping_bwd<scalar_t>(
                means.data_ptr<scalar_t>(),
                covars.data_ptr<scalar_t>(),
                ranges.data_ptr<scalar_t>(),
                centers.data_ptr<scalar_t>(),
                dim2, dim1, dim0,
                static_cast<scalar_t>(eps3d),
                v_means3d_v.data_ptr<scalar_t>(),
                v_conics.data_ptr<scalar_t>(),
                v_means.data_ptr<scalar_t>(),
                v_covars.data_ptr<scalar_t>(),
                static_cast<uint32_t>(N),
                stream
            );
        }
    );

    return std::make_tuple(v_means, v_covars);
}

} // namespace r2splat
