/**
 * Isect Tiles CUDA Kernels
 * 
 * This file implements CUDA kernels for computing tile-Gaussian intersections.
 * 
 * The implementation uses a two-pass algorithm:
 * 1. First pass: Count tiles per Gaussian (compute_tiles_per_gauss_kernel)
 * 2. Second pass: Generate intersection records (generate_isect_records_kernel)
 * 
 * Key features:
 * - Split kernels for better GPU efficiency (no branch divergence)
 * - No packed mode: input is always [C, N, ...]
 * - No depth sorting: CT reconstruction uses ray summation
 * - Efficient sorting by (camera_id, tile_id) only
 * 
 * Author: Dy164
 * Date: 2025-01-12
 */

#include "IsectTiles.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

namespace r2splat {

namespace cg = cooperative_groups;

template <typename OpS>
__device__ __forceinline__ uint32_t clamp_tile_bound(OpS value, uint32_t upper) {
    if (!isfinite(value) || value <= static_cast<OpS>(0)) {
        return 0u;
    }
    const OpS upper_f = static_cast<OpS>(upper);
    if (value >= upper_f) {
        return upper;
    }
    return static_cast<uint32_t>(value);
}

// ============================================================================
// First Pass Kernel: Compute tiles per Gaussian
// ============================================================================

/**
 * Compute the number of tiles each Gaussian covers.
 * 
 * Each thread processes one Gaussian, computing the bounding box
 * of tiles it intersects based on its 2D center and radius.
 * 
 * Dimension convention (following RasterizeToPixelsFwd.cu):
 *   - px -- j -- dim1 -- width (x-axis, columns)
 *   - py -- i -- dim0 -- height (y-axis, rows)
 */
template <typename S>
__global__ void compute_tiles_per_gauss_kernel(
    const S* __restrict__ means2d,          // [C * N, 2]
    const int32_t* __restrict__ radii,      // [C * N] or [C * N, 2]
    int32_t* __restrict__ tiles_per_gauss,  // [C * N]
    const uint32_t total_elems,             // C * N
    const bool radii_is_rect,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,            // number of tiles in x (columns)
    const uint32_t n_tiles_dim0             // number of tiles in y (rows)
) {
    // For mixed precision support
    using OpS = typename OpType<S>::type;
    
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= total_elems) {
        return;
    }
    
    // Get radius for this Gaussian. Rectangular radii store (rx, ry).
    const OpS radius_x = static_cast<OpS>(radii_is_rect ? radii[2 * idx] : radii[idx]);
    const OpS radius_y = static_cast<OpS>(radii_is_rect ? radii[2 * idx + 1] : radii[idx]);
    if (radius_x <= 0 || radius_y <= 0) {
        tiles_per_gauss[idx] = 0;
        return;
    }
    
    // Get 2D center
    const OpS mean_x = static_cast<OpS>(means2d[2 * idx]);
    const OpS mean_y = static_cast<OpS>(means2d[2 * idx + 1]);
    if (!isfinite(mean_x) || !isfinite(mean_y)) {
        tiles_per_gauss[idx] = 0;
        return;
    }
    
    // Convert to tile coordinates
    const OpS tile_size_f = static_cast<OpS>(tile_size);
    const OpS tile_radius_x = radius_x / tile_size_f;
    const OpS tile_radius_y = radius_y / tile_size_f;
    const OpS tile_x = mean_x / tile_size_f;
    const OpS tile_y = mean_y / tile_size_f;
    
    // Compute tile bounding box (tile_min inclusive, tile_max exclusive)
    // dim1 = x = columns, dim0 = y = rows
    uint32_t tile_min_dim1 = clamp_tile_bound(floor(tile_x - tile_radius_x), n_tiles_dim1);
    uint32_t tile_min_dim0 = clamp_tile_bound(floor(tile_y - tile_radius_y), n_tiles_dim0);
    uint32_t tile_max_dim1 = clamp_tile_bound(ceil(tile_x + tile_radius_x), n_tiles_dim1);
    uint32_t tile_max_dim0 = clamp_tile_bound(ceil(tile_y + tile_radius_y), n_tiles_dim0);
    if (tile_max_dim1 <= tile_min_dim1 || tile_max_dim0 <= tile_min_dim0) {
        tiles_per_gauss[idx] = 0;
        return;
    }
    
    // Count tiles
    tiles_per_gauss[idx] = static_cast<int32_t>(
        (tile_max_dim0 - tile_min_dim0) * (tile_max_dim1 - tile_min_dim1)
    );
}

// ============================================================================
// Second Pass Kernel: Generate intersection records
// ============================================================================

/**
 * Generate complete intersection records for all Gaussian-tile pairs.
 * 
 * Each thread processes one Gaussian, writing its intersection records
 * to a pre-allocated contiguous region determined by cum_tiles_per_gauss.
 * 
 * The isect_ids encode (camera_id, tile_id) for efficient sorting:
 *   isect_id = camera_id * n_tiles + tile_id
 *   where tile_id = tile_dim0 * n_tiles_dim1 + tile_dim1
 * 
 * This encoding ensures that after sorting:
 * - Primary order: by camera_id
 * - Secondary order: by tile_id within each camera
 * 
 * Dimension convention:
 *   - dim1 = x = columns (tile_id varies fastest)
 *   - dim0 = y = rows
 */
template <typename S>
__global__ void generate_isect_records_kernel(
    const S* __restrict__ means2d,                   // [C * N, 2]
    const int32_t* __restrict__ radii,               // [C * N] or [C * N, 2]
    const int64_t* __restrict__ cum_tiles_per_gauss, // [C * N]
    int64_t* __restrict__ isect_ids,                 // [n_isects]
    int32_t* __restrict__ flatten_ids,               // [n_isects]
    const uint32_t total_elems,                      // C * N
    const uint32_t N,                                // Gaussians per camera
    const bool radii_is_rect,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,                     // number of tiles in x (columns)
    const uint32_t n_tiles_dim0,                     // number of tiles in y (rows)
    const uint32_t n_tiles                           // n_tiles_dim0 * n_tiles_dim1
) {
    // For mixed precision support
    using OpS = typename OpType<S>::type;
    
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= total_elems) {
        return;
    }
    
    // Get radius for this Gaussian. Rectangular radii store (rx, ry).
    const OpS radius_x = static_cast<OpS>(radii_is_rect ? radii[2 * idx] : radii[idx]);
    const OpS radius_y = static_cast<OpS>(radii_is_rect ? radii[2 * idx + 1] : radii[idx]);
    if (radius_x <= 0 || radius_y <= 0) {
        return;  // No intersections
    }
    
    // Get 2D center
    const OpS mean_x = static_cast<OpS>(means2d[2 * idx]);
    const OpS mean_y = static_cast<OpS>(means2d[2 * idx + 1]);
    if (!isfinite(mean_x) || !isfinite(mean_y)) {
        return;
    }
    
    // Convert to tile coordinates
    const OpS tile_size_f = static_cast<OpS>(tile_size);
    const OpS tile_radius_x = radius_x / tile_size_f;
    const OpS tile_radius_y = radius_y / tile_size_f;
    const OpS tile_x = mean_x / tile_size_f;
    const OpS tile_y = mean_y / tile_size_f;
    
    // Compute tile bounding box (same as first pass)
    // dim1 = x = columns, dim0 = y = rows
    uint32_t tile_min_dim1 = clamp_tile_bound(floor(tile_x - tile_radius_x), n_tiles_dim1);
    uint32_t tile_min_dim0 = clamp_tile_bound(floor(tile_y - tile_radius_y), n_tiles_dim0);
    uint32_t tile_max_dim1 = clamp_tile_bound(ceil(tile_x + tile_radius_x), n_tiles_dim1);
    uint32_t tile_max_dim0 = clamp_tile_bound(ceil(tile_y + tile_radius_y), n_tiles_dim0);
    if (tile_max_dim1 <= tile_min_dim1 || tile_max_dim0 <= tile_min_dim0) {
        return;
    }
    
    // Compute camera ID from flattened index
    // idx = cid * N + gid, so cid = idx / N
    const int64_t cid = static_cast<int64_t>(idx / N);
    
    // Get write offset from cumulative sum
    // (idx == 0) means start at 0, otherwise read from previous element
    int64_t write_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    
    // Write intersection records for each tile
    // Iterate over dim0 (rows) then dim1 (columns)
    for (uint32_t t_dim0 = tile_min_dim0; t_dim0 < tile_max_dim0; ++t_dim0) {
        for (uint32_t t_dim1 = tile_min_dim1; t_dim1 < tile_max_dim1; ++t_dim1) {
            // Tile ID within this camera's tile grid
            // tile_id = dim0 * n_tiles_dim1 + dim1 (row-major order)
            const int64_t tile_id = static_cast<int64_t>(t_dim0 * n_tiles_dim1 + t_dim1);
            
            // Encode: camera_id * n_tiles + tile_id
            // This ensures sorting by (camera_id, tile_id)
            isect_ids[write_idx] = cid * static_cast<int64_t>(n_tiles) + tile_id;
            
            // Store the original flattened Gaussian index
            flatten_ids[write_idx] = static_cast<int32_t>(idx);
            
            ++write_idx;
        }
    }
}

// ============================================================================
// Offset Encoding Kernel
// ============================================================================

/**
 * Encode tile offsets from sorted isect_ids.
 * 
 * After sorting, this kernel determines the starting index for each
 * (camera, tile) pair in the sorted arrays.
 * 
 * Algorithm:
 * - Each thread looks at its isect_id and the previous one
 * - If they differ, fill offsets for the gap between them
 * - Handle boundary cases (first and last elements)
 */
__global__ void isect_offset_encode_kernel(
    const int64_t* __restrict__ isect_ids,  // [n_isects]
    int32_t* __restrict__ offsets,          // [C * n_tiles]
    const uint32_t n_isects,
    const uint32_t C,
    const uint32_t n_tiles
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects) {
        return;
    }
    
    // Current isect_id encodes: camera_id * n_tiles + tile_id
    const int64_t id_curr = isect_ids[idx];
    
    // Handle first element: fill offsets from 0 to id_curr (inclusive)
    if (idx == 0) {
        for (int64_t i = 0; i <= id_curr; ++i) {
            offsets[i] = 0;
        }
    }
    
    // Handle last element: fill offsets from id_curr + 1 to end
    if (idx == n_isects - 1) {
        for (int64_t i = id_curr + 1; i < static_cast<int64_t>(C * n_tiles); ++i) {
            offsets[i] = static_cast<int32_t>(n_isects);
        }
    }
    
    // Handle transitions between different isect_ids
    if (idx > 0) {
        const int64_t id_prev = isect_ids[idx - 1];
        if (id_prev != id_curr) {
            // Fill offsets for the gap between prev and curr
            for (int64_t i = id_prev + 1; i <= id_curr; ++i) {
                offsets[i] = static_cast<int32_t>(idx);
            }
        }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

template <typename S>
void launch_compute_tiles_per_gauss(
    const S* __restrict__ means2d,
    const int32_t* __restrict__ radii,
    int32_t* __restrict__ tiles_per_gauss,
    uint32_t C,
    uint32_t N,
    bool radii_is_rect,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    const uint32_t total_elems = C * N;
    if (total_elems == 0) return;
    
    const uint32_t n_blocks = (total_elems + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    compute_tiles_per_gauss_kernel<S><<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        means2d,
        radii,
        tiles_per_gauss,
        total_elems,
        radii_is_rect,
        tile_size,
        n_tiles_dim1,
        n_tiles_dim0
    );
}

template <typename S>
void launch_generate_isect_records(
    const S* __restrict__ means2d,
    const int32_t* __restrict__ radii,
    const int64_t* __restrict__ cum_tiles_per_gauss,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ flatten_ids,
    uint32_t C,
    uint32_t N,
    bool radii_is_rect,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    uint32_t n_tiles,
    cudaStream_t stream
) {
    const uint32_t total_elems = C * N;
    if (total_elems == 0) return;
    
    const uint32_t n_blocks = (total_elems + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    generate_isect_records_kernel<S><<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        means2d,
        radii,
        cum_tiles_per_gauss,
        isect_ids,
        flatten_ids,
        total_elems,
        N,
        radii_is_rect,
        tile_size,
        n_tiles_dim1,
        n_tiles_dim0,
        n_tiles
    );
}

void launch_isect_offset_encode(
    const int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ offsets,
    uint32_t n_isects,
    uint32_t C,
    uint32_t n_tiles,
    uint32_t tile_n_bits,
    cudaStream_t stream
) {
    if (n_isects == 0) return;
    
    const uint32_t n_blocks = (n_isects + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    isect_offset_encode_kernel<<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        isect_ids,
        offsets,
        n_isects,
        C,
        n_tiles
    );
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_compute_tiles_per_gauss<float>(
    const float*, const int32_t*, int32_t*,
    uint32_t, uint32_t, bool, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_compute_tiles_per_gauss<double>(
    const double*, const int32_t*, int32_t*,
    uint32_t, uint32_t, bool, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_generate_isect_records<float>(
    const float*, const int32_t*, const int64_t*,
    int64_t*, int32_t*,
    uint32_t, uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_generate_isect_records<double>(
    const double*, const int32_t*, const int64_t*,
    int64_t*, int32_t*,
    uint32_t, uint32_t, bool, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

// ============================================================================
// Radix Sort for Intersection IDs (using CUB)
// ============================================================================

std::pair<bool, bool> launch_radix_sort_pairs(
    int64_t* isect_ids_in,
    int64_t* isect_ids_out,
    int32_t* flatten_ids_in,
    int32_t* flatten_ids_out,
    int64_t n_isects,
    uint32_t end_bit,
    cudaStream_t stream
) {
    if (n_isects == 0) {
        return std::make_pair(false, false);
    }
    
    // Create DoubleBuffer wrappers
    cub::DoubleBuffer<int64_t> d_keys(isect_ids_in, isect_ids_out);
    cub::DoubleBuffer<int32_t> d_values(flatten_ids_in, flatten_ids_out);
    
    // Determine temporary storage requirements
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        d_keys, d_values,
        static_cast<int>(n_isects),
        0, end_bit,
        stream
    );
    
    // Allocate temporary storage
    void* d_temp_storage = nullptr;
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys, d_values,
        static_cast<int>(n_isects),
        0, end_bit,
        stream
    );
    
    // Free temporary storage
    cudaFree(d_temp_storage);
    
    // Return which buffers contain the sorted result
    // selector == 0 means result is in the *_in buffer (first buffer)
    // selector == 1 means result is in the *_out buffer (second buffer)
    return std::make_pair(d_keys.selector == 0, d_values.selector == 0);
}

}  // namespace r2splat
