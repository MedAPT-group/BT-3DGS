/**
 * RasterizeToPixels C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Author: Dy164
 * Date: 2025-01-22
 */

#include "RasterizeToPixels.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// Forward Pass Tensor Interface
// ============================================================================

torch::Tensor rasterize_to_pixels_fwd_tensor(
    const torch::Tensor& means2d,       // [C, N, 2]
    const torch::Tensor& conics,        // [C, N, 3]
    const torch::Tensor& opacities,     // [C, N]
    const torch::Tensor& mus,           // [C, N]
    const torch::Tensor& tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const torch::Tensor& flatten_ids,   // [n_isects]
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
) {
    R2SPLAT_DEVICE_GUARD(means2d);
    R2SPLAT_CHECK_INPUT(means2d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(mus);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(flatten_ids);

    // Validate input dimensions
    TORCH_CHECK(means2d.dim() == 3, "means2d must be 3D [C, N, 2]");
    TORCH_CHECK(conics.dim() == 3, "conics must be 3D [C, N, 3]");
    TORCH_CHECK(opacities.dim() == 2, "opacities must be 2D [C, N]");
    TORCH_CHECK(mus.dim() == 2, "mus must be 2D [C, N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D [C, n_tiles_dim0, n_tiles_dim1]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D [n_isects]");

    TORCH_CHECK(means2d.size(2) == 2, "means2d last dim must be 2");
    TORCH_CHECK(conics.size(2) == 3, "conics last dim must be 3");

    // Extract dimensions
    const uint32_t C = static_cast<uint32_t>(means2d.size(0));
    const uint32_t N = static_cast<uint32_t>(means2d.size(1));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(2));

    // Validate dimension consistency
    TORCH_CHECK(conics.size(0) == C && conics.size(1) == N,
                "conics shape must match means2d [C, N, 3]");
    TORCH_CHECK(opacities.size(0) == C && opacities.size(1) == N,
                "opacities shape must match [C, N]");
    TORCH_CHECK(mus.size(0) == C && mus.size(1) == N,
                "mus shape must match [C, N]");
    TORCH_CHECK(tile_offsets.size(0) == C,
                "tile_offsets C dimension must match");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Allocate output tensor
    torch::Tensor render_output = torch::empty(
        {static_cast<int64_t>(C), 
         static_cast<int64_t>(dim0), 
         static_cast<int64_t>(dim1)},
        means2d.options()
    );

    // Handle empty case
    if (C == 0 || N == 0 || n_isects == 0) {
        render_output.zero_();
        return render_output;
    }

    // Dispatch to CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "rasterize_to_pixels_fwd",
        [&]() {
            launch_rasterize_to_pixels_fwd<scalar_t>(
                means2d.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                opacities.data_ptr<scalar_t>(),
                mus.data_ptr<scalar_t>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_output.data_ptr<scalar_t>(),
                C,
                N,
                n_isects,
                dim1,
                dim0,
                tile_size,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );

    return render_output;
}

// ============================================================================
// Backward Pass Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_tensor(
    const torch::Tensor& means2d,       // [C, N, 2]
    const torch::Tensor& conics,        // [C, N, 3]
    const torch::Tensor& opacities,     // [C, N]
    const torch::Tensor& mus,           // [C, N]
    const torch::Tensor& tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const torch::Tensor& flatten_ids,   // [n_isects]
    const torch::Tensor& v_render_output,  // [C, dim0, dim1]
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const bool absgrad
) {
    R2SPLAT_DEVICE_GUARD(means2d);
    R2SPLAT_CHECK_INPUT(means2d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(mus);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(flatten_ids);
    R2SPLAT_CHECK_INPUT(v_render_output);

    // Validate input dimensions
    TORCH_CHECK(means2d.dim() == 3, "means2d must be 3D [C, N, 2]");
    TORCH_CHECK(conics.dim() == 3, "conics must be 3D [C, N, 3]");
    TORCH_CHECK(opacities.dim() == 2, "opacities must be 2D [C, N]");
    TORCH_CHECK(mus.dim() == 2, "mus must be 2D [C, N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D [C, n_tiles_dim0, n_tiles_dim1]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D [n_isects]");
    TORCH_CHECK(v_render_output.dim() == 3, "v_render_output must be 3D [C, dim0, dim1]");

    TORCH_CHECK(means2d.size(2) == 2, "means2d last dim must be 2");
    TORCH_CHECK(conics.size(2) == 3, "conics last dim must be 3");

    // Extract dimensions
    const uint32_t C = static_cast<uint32_t>(means2d.size(0));
    const uint32_t N = static_cast<uint32_t>(means2d.size(1));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(2));

    // Validate dimension consistency
    TORCH_CHECK(conics.size(0) == C && conics.size(1) == N,
                "conics shape must match means2d [C, N, 3]");
    TORCH_CHECK(opacities.size(0) == C && opacities.size(1) == N,
                "opacities shape must match [C, N]");
    TORCH_CHECK(mus.size(0) == C && mus.size(1) == N,
                "mus shape must match [C, N]");
    TORCH_CHECK(tile_offsets.size(0) == C,
                "tile_offsets C dimension must match");
    TORCH_CHECK(v_render_output.size(0) == C && 
                v_render_output.size(1) == static_cast<int64_t>(dim0) &&
                v_render_output.size(2) == static_cast<int64_t>(dim1),
                "v_render_output shape must be [C, dim0, dim1]");

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

    // Allocate output gradient tensors (initialize to zero for atomic add)
    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_means2d_abs = absgrad ? torch::zeros_like(means2d) : torch::Tensor();
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_mus = torch::zeros_like(mus);

    // Handle empty case
    if (C == 0 || N == 0 || n_isects == 0) {
        return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_opacities, v_mus);
    }

    // Dispatch to CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "rasterize_to_pixels_bwd",
        [&]() {
            launch_rasterize_to_pixels_bwd<scalar_t>(
                means2d.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                opacities.data_ptr<scalar_t>(),
                mus.data_ptr<scalar_t>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                v_render_output.data_ptr<scalar_t>(),
                absgrad ? v_means2d_abs.data_ptr<scalar_t>() : nullptr,
                v_means2d.data_ptr<scalar_t>(),
                v_conics.data_ptr<scalar_t>(),
                v_opacities.data_ptr<scalar_t>(),
                v_mus.data_ptr<scalar_t>(),
                C,
                N,
                n_isects,
                dim1,
                dim0,
                tile_size,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_opacities, v_mus);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
rasterize_to_pixels_bwd_per_gaussian_tensor(
    const torch::Tensor& means2d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& mus,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& isect_ids,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const bool absgrad
) {
    R2SPLAT_DEVICE_GUARD(means2d);
    R2SPLAT_CHECK_INPUT(means2d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(mus);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(isect_ids);
    R2SPLAT_CHECK_INPUT(flatten_ids);
    R2SPLAT_CHECK_INPUT(v_render_output);

    TORCH_CHECK(means2d.dim() == 3, "means2d must be 3D [C, N, 2]");
    TORCH_CHECK(conics.dim() == 3, "conics must be 3D [C, N, 3]");
    TORCH_CHECK(opacities.dim() == 2, "opacities must be 2D [C, N]");
    TORCH_CHECK(mus.dim() == 2, "mus must be 2D [C, N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D [C, n_tiles_dim0, n_tiles_dim1]");
    TORCH_CHECK(isect_ids.dim() == 1, "isect_ids must be 1D [n_isects]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D [n_isects]");
    TORCH_CHECK(v_render_output.dim() == 3, "v_render_output must be 3D [C, dim0, dim1]");
    TORCH_CHECK(isect_ids.scalar_type() == torch::kInt64, "isect_ids must be int64");

    const uint32_t C = static_cast<uint32_t>(means2d.size(0));
    const uint32_t N = static_cast<uint32_t>(means2d.size(1));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(2));

    TORCH_CHECK(isect_ids.size(0) == flatten_ids.size(0), "isect_ids and flatten_ids must have the same length");
    TORCH_CHECK(means2d.size(2) == 2, "means2d last dim must be 2");
    TORCH_CHECK(conics.size(0) == C && conics.size(1) == N && conics.size(2) == 3, "conics shape must match [C, N, 3]");
    TORCH_CHECK(opacities.size(0) == C && opacities.size(1) == N, "opacities shape must match [C, N]");
    TORCH_CHECK(mus.size(0) == C && mus.size(1) == N, "mus shape must match [C, N]");
    TORCH_CHECK(tile_offsets.size(0) == C, "tile_offsets C dimension must match");
    TORCH_CHECK(
        v_render_output.size(0) == C &&
        v_render_output.size(1) == static_cast<int64_t>(dim0) &&
        v_render_output.size(2) == static_cast<int64_t>(dim1),
        "v_render_output shape must be [C, dim0, dim1]"
    );

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_means2d_abs = absgrad ? torch::zeros_like(means2d) : torch::Tensor();
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_mus = torch::zeros_like(mus);

    if (C == 0 || N == 0 || n_isects == 0) {
        return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_opacities, v_mus);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(means2d.scalar_type(), "rasterize_to_pixels_bwd_per_gaussian", [&]() {
        launch_rasterize_to_pixels_bwd_per_gaussian<scalar_t>(
            means2d.data_ptr<scalar_t>(),
            conics.data_ptr<scalar_t>(),
            opacities.data_ptr<scalar_t>(),
            mus.data_ptr<scalar_t>(),
            isect_ids.data_ptr<int64_t>(),
            flatten_ids.data_ptr<int32_t>(),
            v_render_output.data_ptr<scalar_t>(),
            absgrad ? v_means2d_abs.data_ptr<scalar_t>() : nullptr,
            v_means2d.data_ptr<scalar_t>(),
            v_conics.data_ptr<scalar_t>(),
            v_opacities.data_ptr<scalar_t>(),
            v_mus.data_ptr<scalar_t>(),
            C,
            N,
            n_isects,
            dim1,
            dim0,
            tile_size,
            n_tiles_dim1,
            n_tiles_dim0,
            stream
        );
    });

    return std::make_tuple(v_means2d_abs, v_means2d, v_conics, v_opacities, v_mus);
}

} // namespace r2splat

