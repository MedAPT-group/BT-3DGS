/**
 * Isect Tiles C++ Wrapper
 * 
 * This file provides the C++ tensor wrapper functions that:
 * 1. Validate input tensors (CUDA, contiguous, shape)
 * 2. Allocate output tensors
 * 3. Dispatch to type-specific CUDA kernels
 * 4. Handle sorting and offset encoding
 * 
 * Following gsplat's pattern for tensor handling.
 * 
 * Note: CUB operations are in IsectTilesKernel.cu because CUB requires CUDA compilation.
 * 
 * Author: Dy164
 * Date: 2025-01-12
 */

#include "IsectTiles.h"
#include "../include/Ops.h"
#include <torch/extension.h>
#include <tuple>
#include <cmath>

namespace r2splat {

// ============================================================================
// Isect Tiles Tensor Interface
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
isect_tiles_tensor(
    const torch::Tensor& means2d,    // [C, N, 2]
    const torch::Tensor& radii,      // [C, N] or [C, N, 2]
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,     // number of tiles in x (columns)
    const uint32_t n_tiles_dim0      // number of tiles in y (rows)
) {
    R2SPLAT_DEVICE_GUARD(means2d);
    R2SPLAT_CHECK_INPUT(means2d);
    R2SPLAT_CHECK_INPUT(radii);
    
    // Validate input dimensions
    TORCH_CHECK(means2d.dim() == 3, "means2d must be 3D [C, N, 2]");
    TORCH_CHECK(
        radii.dim() == 2 || (radii.dim() == 3 && radii.size(2) == 2),
        "radii must be 2D [C, N] or 3D [C, N, 2]"
    );
    TORCH_CHECK(means2d.size(2) == 2, "means2d last dim must be 2");
    TORCH_CHECK(means2d.size(0) == radii.size(0), "C dimension must match");
    TORCH_CHECK(means2d.size(1) == radii.size(1), "N dimension must match");
    
    const uint32_t C = static_cast<uint32_t>(means2d.size(0));
    const uint32_t N = static_cast<uint32_t>(means2d.size(1));
    const uint32_t total_elems = C * N;
    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1;
    const bool radii_is_rect = radii.dim() == 3;
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // Allocate tiles_per_gauss
    torch::Tensor tiles_per_gauss = torch::empty(
        {C, N}, radii.options().dtype(torch::kInt32)
    );
    
    if (total_elems == 0) {
        // Return empty tensors
        torch::Tensor isect_ids = torch::empty({0}, radii.options().dtype(torch::kInt64));
        torch::Tensor flatten_ids = torch::empty({0}, radii.options().dtype(torch::kInt32));
        return std::make_tuple(tiles_per_gauss, isect_ids, flatten_ids);
    }
    
    // ========================================================================
    // First Pass: Compute tiles per Gaussian
    // ========================================================================
    
    AT_DISPATCH_FLOATING_TYPES(
        means2d.scalar_type(),
        "compute_tiles_per_gauss",
        [&]() {
            launch_compute_tiles_per_gauss<scalar_t>(
                means2d.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                tiles_per_gauss.data_ptr<int32_t>(),
                C, N,
                radii_is_rect,
                tile_size,
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
        means2d.scalar_type(),
        "generate_isect_records",
        [&]() {
            launch_generate_isect_records<scalar_t>(
                means2d.data_ptr<scalar_t>(),
                radii.data_ptr<int32_t>(),
                cum_tiles_per_gauss.data_ptr<int64_t>(),
                isect_ids.data_ptr<int64_t>(),
                flatten_ids.data_ptr<int32_t>(),
                C, N,
                radii_is_rect,
                tile_size,
                n_tiles_dim1,
                n_tiles_dim0,
                n_tiles,
                stream
            );
        }
    );
    
    // ========================================================================
    // Sorting by (camera_id, tile_id) - Always required for rasterization
    // ========================================================================
    
    torch::Tensor isect_ids_sorted = torch::empty_like(isect_ids);
    torch::Tensor flatten_ids_sorted = torch::empty_like(flatten_ids);
    
    // Sort by all bits of isect_id (camera_id * n_tiles + tile_id)
    // We only need enough bits to represent C * n_tiles
    uint32_t max_id = C * n_tiles;
    uint32_t end_bit = static_cast<uint32_t>(ceil(log2(static_cast<double>(max_id + 1))));
    
    // Call CUDA sorting function (CUB is only available in .cu files)
    auto [keys_in_place, values_in_place] = launch_radix_sort_pairs(
        isect_ids.data_ptr<int64_t>(),
        isect_ids_sorted.data_ptr<int64_t>(),
        flatten_ids.data_ptr<int32_t>(),
        flatten_ids_sorted.data_ptr<int32_t>(),
        n_isects,
        end_bit,
        stream
    );
    
    // Determine which buffer has the sorted result
    // If *_in_place is true, the result is in the original (*_in) buffer
    if (keys_in_place) {
        isect_ids_sorted = isect_ids;
    }
    if (values_in_place) {
        flatten_ids_sorted = flatten_ids;
    }
    
    return std::make_tuple(tiles_per_gauss, isect_ids_sorted, flatten_ids_sorted);
}

// ============================================================================
// Isect Offset Encode Tensor Interface
// ============================================================================

torch::Tensor isect_offset_encode_tensor(
    const torch::Tensor& isect_ids,  // [n_isects]
    const uint32_t C,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0
) {
    R2SPLAT_DEVICE_GUARD(isect_ids);
    R2SPLAT_CHECK_INPUT(isect_ids);
    
    const uint32_t n_isects = static_cast<uint32_t>(isect_ids.size(0));
    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1;
    
    // Allocate offsets tensor with shape [C, n_tiles_dim0, n_tiles_dim1]
    torch::Tensor offsets = torch::empty(
        {static_cast<int64_t>(C), static_cast<int64_t>(n_tiles_dim0), static_cast<int64_t>(n_tiles_dim1)},
        isect_ids.options().dtype(torch::kInt32)
    );
    
    if (n_isects == 0) {
        offsets.fill_(0);
        return offsets;
    }
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    
    // tile_n_bits is not used in our simplified encoding, but kept for API compatibility
    uint32_t tile_n_bits = static_cast<uint32_t>(ceil(log2(static_cast<double>(n_tiles + 1))));
    
    launch_isect_offset_encode(
        isect_ids.data_ptr<int64_t>(),
        offsets.data_ptr<int32_t>(),
        n_isects,
        C,
        n_tiles,
        tile_n_bits,
        stream
    );
    
    return offsets;
}

}  // namespace r2splat
