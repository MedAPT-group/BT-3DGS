/**
 * VoxelizeToVoxels C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Dimension convention:
 *   - dim2 = x (fastest varying in memory)
 *   - dim1 = y (middle dimension)
 *   - dim0 = z (slowest varying in memory)
 *   - Output shape: [dim0, dim1, dim2]
 * 
 * Author: Dy164
 * Date: 2026-02-02
 */

#include "VoxelizeToVoxels.h"
#include "../include/Ops.h"
#include <torch/extension.h>

namespace r2splat {

// ============================================================================
// VoxelizeToVoxels Forward Tensor Interface
// ============================================================================

torch::Tensor voxelize_to_voxels_fwd_tensor(
    const torch::Tensor& means3d,       // [N, 3]
    const torch::Tensor& conics,        // [N, 6]
    const torch::Tensor& opacities,     // [N]
    const torch::Tensor& tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const torch::Tensor& flatten_ids,   // [n_isects]
    const uint32_t dim2,                // x size (fastest)
    const uint32_t dim1,                // y size (middle)
    const uint32_t dim0,                // z size (slowest)
    const uint32_t tile_size
) {
    R2SPLAT_DEVICE_GUARD(means3d);
    R2SPLAT_CHECK_INPUT(means3d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(flatten_ids);
    
    // Validate input dimensions
    TORCH_CHECK(means3d.dim() == 2, "means3d must be 2D [N, 3]");
    TORCH_CHECK(conics.dim() == 2, "conics must be 2D [N, 6]");
    TORCH_CHECK(opacities.dim() == 1, "opacities must be 1D [N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D [n_isects]");
    
    TORCH_CHECK(means3d.size(1) == 3, "means3d last dim must be 3");
    TORCH_CHECK(conics.size(1) == 6, "conics last dim must be 6");
    
    const uint32_t N = static_cast<uint32_t>(means3d.size(0));
    TORCH_CHECK(conics.size(0) == static_cast<int64_t>(N), "conics first dim must match N");
    TORCH_CHECK(opacities.size(0) == static_cast<int64_t>(N), "opacities first dim must match N");
    
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(0));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim2 = static_cast<uint32_t>(tile_offsets.size(2));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // Allocate output tensor with shape [dim0, dim1, dim2]
    torch::Tensor render_output = torch::zeros(
        {static_cast<int64_t>(dim0), static_cast<int64_t>(dim1), static_cast<int64_t>(dim2)},
        means3d.options()
    );
    
    if (N == 0 || n_isects == 0) {
        // Return zeros if no Gaussians or no intersections
        return render_output;
    }
    
    // Dispatch to CUDA kernel
    AT_DISPATCH_FLOATING_TYPES(
        means3d.scalar_type(),
        "voxelize_to_voxels_fwd",
        [&]() {
            launch_voxelize_to_voxels_fwd<scalar_t>(
                means3d.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                opacities.data_ptr<scalar_t>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                render_output.data_ptr<scalar_t>(),
                N,
                n_isects,
                dim2,
                dim1,
                dim0,
                tile_size,
                n_tiles_dim2,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );
    
    return render_output;
}

// ============================================================================
// VoxelizeToVoxels First-Order Backward Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxelize_to_voxels_bwd_tensor(
    const torch::Tensor& means3d,       // [N, 3]
    const torch::Tensor& conics,        // [N, 6]
    const torch::Tensor& opacities,     // [N]
    const torch::Tensor& tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const torch::Tensor& flatten_ids,   // [n_isects]
    const torch::Tensor& v_render_output,  // [dim0, dim1, dim2]
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
) {
    R2SPLAT_DEVICE_GUARD(means3d);
    R2SPLAT_CHECK_INPUT(means3d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(flatten_ids);
    R2SPLAT_CHECK_INPUT(v_render_output);
    
    // Validate input dimensions
    TORCH_CHECK(means3d.dim() == 2, "means3d must be 2D [N, 3]");
    TORCH_CHECK(conics.dim() == 2, "conics must be 2D [N, 6]");
    TORCH_CHECK(opacities.dim() == 1, "opacities must be 1D [N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");
    TORCH_CHECK(v_render_output.dim() == 3, "v_render_output must be 3D [dim0, dim1, dim2]");
    
    TORCH_CHECK(means3d.size(1) == 3, "means3d last dim must be 3");
    TORCH_CHECK(conics.size(1) == 6, "conics last dim must be 6");
    
    const uint32_t N = static_cast<uint32_t>(means3d.size(0));
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(0));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim2 = static_cast<uint32_t>(tile_offsets.size(2));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // Allocate output gradient tensors
    torch::Tensor v_means3d = torch::zeros_like(means3d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    
    if (N == 0 || n_isects == 0) {
        return std::make_tuple(v_means3d, v_conics, v_opacities);
    }
    
    AT_DISPATCH_FLOATING_TYPES(
        means3d.scalar_type(),
        "voxelize_to_voxels_bwd",
        [&]() {
            launch_voxelize_to_voxels_bwd<scalar_t>(
                means3d.data_ptr<scalar_t>(),
                conics.data_ptr<scalar_t>(),
                opacities.data_ptr<scalar_t>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                v_render_output.data_ptr<scalar_t>(),
                v_means3d.data_ptr<scalar_t>(),
                v_conics.data_ptr<scalar_t>(),
                v_opacities.data_ptr<scalar_t>(),
                N,
                n_isects,
                dim2,
                dim1,
                dim0,
                tile_size,
                n_tiles_dim2,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );
    
    return std::make_tuple(v_means3d, v_conics, v_opacities);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
voxelize_to_voxels_bwd_per_gaussian_tensor(
    const torch::Tensor& means3d,
    const torch::Tensor& conics,
    const torch::Tensor& opacities,
    const torch::Tensor& tile_offsets,
    const torch::Tensor& isect_ids,
    const torch::Tensor& flatten_ids,
    const torch::Tensor& v_render_output,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size
) {
    R2SPLAT_DEVICE_GUARD(means3d);
    R2SPLAT_CHECK_INPUT(means3d);
    R2SPLAT_CHECK_INPUT(conics);
    R2SPLAT_CHECK_INPUT(opacities);
    R2SPLAT_CHECK_INPUT(tile_offsets);
    R2SPLAT_CHECK_INPUT(isect_ids);
    R2SPLAT_CHECK_INPUT(flatten_ids);
    R2SPLAT_CHECK_INPUT(v_render_output);

    TORCH_CHECK(means3d.dim() == 2, "means3d must be 2D [N, 3]");
    TORCH_CHECK(conics.dim() == 2, "conics must be 2D [N, 6]");
    TORCH_CHECK(opacities.dim() == 1, "opacities must be 1D [N]");
    TORCH_CHECK(tile_offsets.dim() == 3, "tile_offsets must be 3D");
    TORCH_CHECK(isect_ids.dim() == 1, "isect_ids must be 1D");
    TORCH_CHECK(flatten_ids.dim() == 1, "flatten_ids must be 1D");
    TORCH_CHECK(v_render_output.dim() == 3, "v_render_output must be 3D [dim0, dim1, dim2]");
    TORCH_CHECK(isect_ids.scalar_type() == torch::kInt64, "isect_ids must be int64");
    TORCH_CHECK(means3d.size(1) == 3, "means3d last dim must be 3");
    TORCH_CHECK(conics.size(1) == 6, "conics last dim must be 6");
    TORCH_CHECK(isect_ids.size(0) == flatten_ids.size(0), "isect_ids and flatten_ids must have the same length");

    const uint32_t N = static_cast<uint32_t>(means3d.size(0));
    const uint32_t n_tiles_dim0 = static_cast<uint32_t>(tile_offsets.size(0));
    const uint32_t n_tiles_dim1 = static_cast<uint32_t>(tile_offsets.size(1));
    const uint32_t n_tiles_dim2 = static_cast<uint32_t>(tile_offsets.size(2));
    const uint32_t n_isects = static_cast<uint32_t>(flatten_ids.size(0));

    TORCH_CHECK(conics.size(0) == static_cast<int64_t>(N), "conics first dim must match N");
    TORCH_CHECK(opacities.size(0) == static_cast<int64_t>(N), "opacities first dim must match N");
    TORCH_CHECK(
        v_render_output.size(0) == static_cast<int64_t>(dim0) &&
        v_render_output.size(1) == static_cast<int64_t>(dim1) &&
        v_render_output.size(2) == static_cast<int64_t>(dim2),
        "v_render_output shape must be [dim0, dim1, dim2]"
    );

    torch::Tensor v_means3d = torch::zeros_like(means3d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_opacities = torch::zeros_like(opacities);

    if (N == 0 || n_isects == 0) {
        return std::make_tuple(v_means3d, v_conics, v_opacities);
    }

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(means3d.scalar_type(), "voxelize_to_voxels_bwd_per_gaussian", [&]() {
        launch_voxelize_to_voxels_bwd_per_gaussian<scalar_t>(
            means3d.data_ptr<scalar_t>(),
            conics.data_ptr<scalar_t>(),
            opacities.data_ptr<scalar_t>(),
            isect_ids.data_ptr<int64_t>(),
            flatten_ids.data_ptr<int32_t>(),
            v_render_output.data_ptr<scalar_t>(),
            v_means3d.data_ptr<scalar_t>(),
            v_conics.data_ptr<scalar_t>(),
            v_opacities.data_ptr<scalar_t>(),
            N,
            n_isects,
            dim2,
            dim1,
            dim0,
            tile_size,
            n_tiles_dim2,
            n_tiles_dim1,
            n_tiles_dim0,
            stream
        );
    });

    return std::make_tuple(v_means3d, v_conics, v_opacities);
}

} // namespace r2splat

