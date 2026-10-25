/**
 * Isect Tiles 3D C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 4. Handle sorting and offset encoding
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Note: CUB operations are in IsectTiles3DKernel.cu because CUB requires CUDA compilation.
 * 
 * Author: Dy164
 * Date: 2025-01-12
 * Updated: 2026-02-02 (Refactored to split kernel pattern, removed depth)
 */

#include "IsectTiles3D.h"
#include "../include/Ops.h"
#include <torch/extension.h>
#include <tuple>
#include <cmath>

namespace r2splat {

// ============================================================================
// Isect Tiles 3D Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_3d_tensor(
    const torch::Tensor& means3d,    // [N, 3]
    const torch::Tensor& radii,      // [N, 3]
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,     // number of tiles in x (fastest)
    const uint32_t n_tiles_dim1,     // number of tiles in y (middle)
    const uint32_t n_tiles_dim0      // number of tiles in z (slowest)
) {
    R2SPLAT_DEVICE_GUARD(means3d);
    R2SPLAT_CHECK_INPUT(means3d);
    R2SPLAT_CHECK_INPUT(radii);
    
    // Validate input dimensions
    TORCH_CHECK(means3d.dim() == 2, "means3d must be 2D [N, 3]");
    TORCH_CHECK(radii.dim() == 2, "radii must be 2D [N, 3]");
    TORCH_CHECK(means3d.size(1) == 3, "means3d last dim must be 3");
    TORCH_CHECK(radii.size(1) == 3, "radii last dim must be 3");
    TORCH_CHECK(means3d.size(0) == radii.size(0), "N dimension must match");
    
    const uint32_t N = static_cast<uint32_t>(means3d.size(0));
    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2;
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // Allocate tiles_per_gauss
    torch::Tensor tiles_per_gauss = torch::empty(
        {static_cast<int64_t>(N)}, radii.options().dtype(torch::kInt32)
    );
    
    if (N == 0) {
        // Return empty tensors
        torch::Tensor isect_ids = torch::empty({0}, radii.options().dtype(torch::kInt64));
        torch::Tensor flatten_ids = torch::empty({0}, radii.options().dtype(torch::kInt32));
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
    
    // ========================================================================
    // First Pass: Compute tiles per Gaussian
    // ========================================================================
    
    AT_DISPATCH_FLOATING_TYPES(
        means3d.scalar_type(),
        "compute_tiles_per_gauss_3d",
        [&]() {
            launch_compute_tiles_per_gauss_3d<scalar_t>(
                means3d.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                tiles_per_gauss.data_ptr<int32_t>(),
                N,
                tile_size,
                n_tiles_dim2,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );
    
    // Compute cumulative sum for write offsets
    torch::Tensor cum_tiles_per_gauss = torch::cumsum(
        tiles_per_gauss.view({-1}), 0, torch::kInt64
    );
    
    // Total number of intersections
    int64_t n_isects = cum_tiles_per_gauss[-1].item<int64_t>();
    
    if (n_isects == 0) {
        // No intersections
        torch::Tensor isect_ids = torch::empty({0}, radii.options().dtype(torch::kInt64));
        torch::Tensor flatten_ids = torch::empty({0}, radii.options().dtype(torch::kInt32));
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
    
    // ========================================================================
    // Second Pass: Generate intersection records
    // ========================================================================
    
    torch::Tensor isect_ids = torch::empty({n_isects}, radii.options().dtype(torch::kInt64));
    torch::Tensor flatten_ids = torch::empty({n_isects}, radii.options().dtype(torch::kInt32));
    
    AT_DISPATCH_FLOATING_TYPES(
        means3d.scalar_type(),
        "generate_isect_records_3d",
        [&]() {
            launch_generate_isect_records_3d<scalar_t>(
                means3d.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                cum_tiles_per_gauss.data_ptr<int64_t>(),
                isect_ids.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                N,
                tile_size,
                n_tiles_dim2,
                n_tiles_dim1,
                n_tiles_dim0,
                stream
            );
        }
    );
    
    // ========================================================================
    // Sorting by tile_id - Always required for rasterization
    // ========================================================================
    
    torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
    torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);
    
    // Sort by all bits of isect_id (tile_id only, no depth)
    uint32_t end_bit = static_cast<uint32_t>(ceil(log2(static_cast<double>(n_tiles + 1))));
    
    // Call CUDA sorting function (CUB is only available in .cu files)
    auto [keys_in_place, values_in_place] = launch_radix_sort_pairs_3d(
        isect_ids.data_ptr<int64_t>(),
        isect_ids_sorted.data_ptr<int64_t>(),
        flatten_ids.data_ptr<int32_t>(),
        flatten_ids_sorted.data_ptr<int32_t>(),
        n_isects,
        end_bit,
        stream
    );
    
    // Determine which buffer has the sorted result
    if (keys_in_place) {
        isect_ids_sorted = isect_ids;
    }
    if (values_in_place) {
        flatten_ids_sorted = flatten_ids;
    }
    
    return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
}

// ============================================================================
// Isect Offset Encode 3D Tensor Interface
// ============================================================================

torch::Tensor isect_offset_encode_3d_tensor(
    const torch::Tensor& isect_ids,  // [n_isects]
    const uint32_t n_tiles_dim2,     // number of tiles in x (fastest)
    const uint32_t n_tiles_dim1,     // number of tiles in y (middle)
    const uint32_t n_tiles_dim0      // number of tiles in z (slowest)
) {
    R2SPLAT_DEVICE_GUARD(isect_ids);
    R2SPLAT_CHECK_INPUT(isect_ids);
    
    const uint32_t n_isects = static_cast<uint32_t>(isect_ids.size(0));
    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2;
    
    // Allocate offsets tensor with shape [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    torch::Tensor offsets = torch::empty(
        {static_cast<int64_t>(n_tiles_dim0), 
         static_cast<int64_t>(n_tiles_dim1), 
         static_cast<int64_t>(n_tiles_dim2)},
        isect_ids.options().dtype(torch::kInt32)
    );
    
    if (n_isects == 0) {
        offsets.fill_(0);
        return offsets;
    }
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    launch_isect_offset_encode_3d(
        isect_ids.data_ptr<int64_t>(),
        offsets.data_ptr<int32_t>(),
        n_isects,
        n_tiles,
        stream
    );
    
    return offsets;
}

}  // namespace r2splat
