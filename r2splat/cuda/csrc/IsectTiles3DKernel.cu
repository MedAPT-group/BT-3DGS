/**
 * Isect Tiles 3D CUDA Kernels
 * 
 * This file implements CUDA kernels for computing tile-Gaussian intersections in 3D.
 * 
 * The implementation uses a two-pass algorithm:
 * 1. First pass: Count tiles per Gaussian (compute_tiles_per_gauss_3d_kernel)
 * 2. Second pass: Generate intersection records (generate_isect_records_3d_kernel)
 * 
 * Key features:
 * - Split kernels for better GPU efficiency (no branch divergence)
 * - No depth sorting: CT reconstruction uses ray summation
 * - Efficient sorting by tile_id only
 * 
 * Refactored from original gsplat IsectTiles3D to match r2splat style.
 * 
 * Author: Dy164
 * Date: 2025-01-12
 * Updated: 2026-02-02 (Refactored to split kernel pattern, removed depth)
 */

#include "IsectTiles3D.h"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <utility>

namespace r2splat {

namespace cg = cooperative_groups;

// ============================================================================
// First Pass Kernel: Compute tiles per Gaussian (3D)
// ============================================================================

/**
 * Compute the number of tiles each Gaussian covers in 3D.
 * 
 * Each thread processes one Gaussian, computing the bounding box
 * of tiles it intersects based on its 3D center and radius.
 * 
 * Dimension convention (following VoxelizeToVoxelsFwd.cu):
 *   - px -- k -- dim2 -- x (fastest varying, columns)
 *   - py -- j -- dim1 -- y (middle dimension)
 *   - pz -- i -- dim0 -- z (slowest varying, rows)
 *   - tile_id = i * (n_tiles_dim1 * n_tiles_dim2) + j * n_tiles_dim2 + k
 */
template <typename S>
__global__ void compute_tiles_per_gauss_3d_kernel(
    const S* __restrict__ means3d,          // [N, 3] stored as (x, y, z)
    const int32_t* __restrict__ radii,      // [N, 3] stored as (rx, ry, rz)
    int32_t* __restrict__ tiles_per_gauss,  // [N]
    const uint32_t N,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,            // number of tiles in x (fastest)
    const uint32_t n_tiles_dim1,            // number of tiles in y (middle)
    const uint32_t n_tiles_dim0             // number of tiles in z (slowest)
) {
    // For mixed precision support
    using OpS = typename OpType<S>::type;
    
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    
    // Get radius for this Gaussian (3D radius)
    const OpS radius_x = static_cast<OpS>(radii[3 * idx]);
    const OpS radius_y = static_cast<OpS>(radii[3 * idx + 1]);
    const OpS radius_z = static_cast<OpS>(radii[3 * idx + 2]);
    
    if (radius_x <= 0 || radius_y <= 0 || radius_z <= 0) {
        tiles_per_gauss[idx] = 0;
        return;
    }
    
    // Get 3D center (x, y, z) = (dim2, dim1, dim0)
    const OpS mean_x = static_cast<OpS>(means3d[3 * idx]);     // dim2
    const OpS mean_y = static_cast<OpS>(means3d[3 * idx + 1]); // dim1
    const OpS mean_z = static_cast<OpS>(means3d[3 * idx + 2]); // dim0
    
    // Convert to tile coordinates
    const OpS tile_size_f = static_cast<OpS>(tile_size);
    const OpS tile_r_x = radius_x / tile_size_f;  // dim2 radius in tiles
    const OpS tile_r_y = radius_y / tile_size_f;  // dim1 radius in tiles
    const OpS tile_r_z = radius_z / tile_size_f;  // dim0 radius in tiles
    const OpS tile_x = mean_x / tile_size_f;      // dim2 center in tiles
    const OpS tile_y = mean_y / tile_size_f;      // dim1 center in tiles
    const OpS tile_z = mean_z / tile_size_f;      // dim0 center in tiles
    
    // Compute tile bounding box (tile_min inclusive, tile_max exclusive)
    // dim2 = x (fastest), dim1 = y (middle), dim0 = z (slowest)
    uint32_t tile_min_dim2 = min(max(0u, static_cast<uint32_t>(floor(tile_x - tile_r_x))), n_tiles_dim2);
    uint32_t tile_min_dim1 = min(max(0u, static_cast<uint32_t>(floor(tile_y - tile_r_y))), n_tiles_dim1);
    uint32_t tile_min_dim0 = min(max(0u, static_cast<uint32_t>(floor(tile_z - tile_r_z))), n_tiles_dim0);
    uint32_t tile_max_dim2 = min(max(0u, static_cast<uint32_t>(ceil(tile_x + tile_r_x))), n_tiles_dim2);
    uint32_t tile_max_dim1 = min(max(0u, static_cast<uint32_t>(ceil(tile_y + tile_r_y))), n_tiles_dim1);
    uint32_t tile_max_dim0 = min(max(0u, static_cast<uint32_t>(ceil(tile_z + tile_r_z))), n_tiles_dim0);
    
    // Count tiles
    tiles_per_gauss[idx] = static_cast<int32_t>(
        (tile_max_dim0 - tile_min_dim0) * (tile_max_dim1 - tile_min_dim1) * (tile_max_dim2 - tile_min_dim2)
    );
}

// ============================================================================
// Second Pass Kernel: Generate intersection records (3D)
// ============================================================================

/**
 * Generate complete intersection records for all Gaussian-tile pairs in 3D.
 * 
 * Each thread processes one Gaussian, writing its intersection records
 * to a pre-allocated contiguous region determined by cum_tiles_per_gauss.
 * 
 * The isect_ids encode tile_id for efficient sorting:
 *   tile_id = dim0 * (n_tiles_dim1 * n_tiles_dim2) + dim1 * n_tiles_dim2 + dim2
 * 
 * Dimension convention:
 *   - dim2 = x (fastest varying in memory)
 *   - dim1 = y (middle dimension)
 *   - dim0 = z (slowest varying in memory)
 */
template <typename S>
__global__ void generate_isect_records_3d_kernel(
    const S* __restrict__ means3d,                   // [N, 3] stored as (x, y, z)
    const int32_t* __restrict__ radii,               // [N, 3] stored as (rx, ry, rz)
    const int64_t* __restrict__ cum_tiles_per_gauss, // [N]
    int64_t* __restrict__ isect_ids,                 // [n_isects]
    int32_t* __restrict__ flatten_ids,               // [n_isects]
    const uint32_t N,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,                     // number of tiles in x (fastest)
    const uint32_t n_tiles_dim1,                     // number of tiles in y (middle)
    const uint32_t n_tiles_dim0                      // number of tiles in z (slowest)
) {
    // For mixed precision support
    using OpS = typename OpType<S>::type;
    
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }
    
    // Get radius for this Gaussian (3D radius)
    const OpS radius_x = static_cast<OpS>(radii[3 * idx]);
    const OpS radius_y = static_cast<OpS>(radii[3 * idx + 1]);
    const OpS radius_z = static_cast<OpS>(radii[3 * idx + 2]);
    
    if (radius_x <= 0 || radius_y <= 0 || radius_z <= 0) {
        return;  // No intersections
    }
    
    // Get 3D center (x, y, z) = (dim2, dim1, dim0)
    const OpS mean_x = static_cast<OpS>(means3d[3 * idx]);     // dim2
    const OpS mean_y = static_cast<OpS>(means3d[3 * idx + 1]); // dim1
    const OpS mean_z = static_cast<OpS>(means3d[3 * idx + 2]); // dim0
    
    // Convert to tile coordinates
    const OpS tile_size_f = static_cast<OpS>(tile_size);
    const OpS tile_r_x = radius_x / tile_size_f;
    const OpS tile_r_y = radius_y / tile_size_f;
    const OpS tile_r_z = radius_z / tile_size_f;
    const OpS tile_x = mean_x / tile_size_f;
    const OpS tile_y = mean_y / tile_size_f;
    const OpS tile_z = mean_z / tile_size_f;
    
    // Compute tile bounding box (same as first pass)
    // dim2 = x (fastest), dim1 = y (middle), dim0 = z (slowest)
    uint32_t tile_min_dim2 = min(max(0u, static_cast<uint32_t>(floor(tile_x - tile_r_x))), n_tiles_dim2);
    uint32_t tile_min_dim1 = min(max(0u, static_cast<uint32_t>(floor(tile_y - tile_r_y))), n_tiles_dim1);
    uint32_t tile_min_dim0 = min(max(0u, static_cast<uint32_t>(floor(tile_z - tile_r_z))), n_tiles_dim0);
    uint32_t tile_max_dim2 = min(max(0u, static_cast<uint32_t>(ceil(tile_x + tile_r_x))), n_tiles_dim2);
    uint32_t tile_max_dim1 = min(max(0u, static_cast<uint32_t>(ceil(tile_y + tile_r_y))), n_tiles_dim1);
    uint32_t tile_max_dim0 = min(max(0u, static_cast<uint32_t>(ceil(tile_z + tile_r_z))), n_tiles_dim0);
    
    // Get write offset from cumulative sum
    int64_t write_idx = (idx == 0) ? 0 : cum_tiles_per_gauss[idx - 1];
    
    // Write intersection records for each tile
    // Iterate: dim0 (z, slowest) -> dim1 (y) -> dim2 (x, fastest)
    for (uint32_t t_dim0 = tile_min_dim0; t_dim0 < tile_max_dim0; ++t_dim0) {
        for (uint32_t t_dim1 = tile_min_dim1; t_dim1 < tile_max_dim1; ++t_dim1) {
            for (uint32_t t_dim2 = tile_min_dim2; t_dim2 < tile_max_dim2; ++t_dim2) {
                // Tile ID: dim0 * (n_tiles_dim1 * n_tiles_dim2) + dim1 * n_tiles_dim2 + dim2
                const int64_t tile_id = static_cast<int64_t>(
                    t_dim0 * n_tiles_dim1 * n_tiles_dim2 + t_dim1 * n_tiles_dim2 + t_dim2
                );
                
                // Store tile_id directly (no depth encoding)
                isect_ids[write_idx] = tile_id;
                
                // Store the original Gaussian index
                flatten_ids[write_idx] = static_cast<int32_t>(idx);
                
                ++write_idx;
            }
        }
    }
}

// ============================================================================
// Offset Encoding Kernel (3D)
// ============================================================================

/**
 * Encode tile offsets from sorted isect_ids for 3D.
 * 
 * After sorting, this kernel determines the starting index for each
 * tile in the sorted arrays.
 */
__global__ void isect_offset_encode_3d_kernel(
    const int64_t* __restrict__ isect_ids,  // [n_isects]
    int32_t* __restrict__ offsets,          // [n_tiles]
    const uint32_t n_isects,
    const uint32_t n_tiles
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= n_isects) {
        return;
    }
    
    // Current isect_id is the tile_id directly
    const int64_t id_curr = isect_ids[idx];
    
    // Handle first element: fill offsets from 0 to id_curr (inclusive)
    if (idx == 0) {
        for (int64_t i = 0; i <= id_curr; ++i) {
            offsets[i] = 0;
        }
    }
    
    // Handle last element: fill offsets from id_curr + 1 to end
    if (idx == n_isects - 1) {
        for (int64_t i = id_curr + 1; i < static_cast<int64_t>(n_tiles); ++i) {
            offsets[i] = static_cast<int32_t>(n_isects);
        }
    }
    
    // Handle transitions between different isect_ids
    if (idx > 0) {
        const int64_t id_prev = isect_ids[idx - 1];
        if (id_prev != id_curr) {
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
void launch_compute_tiles_per_gauss_3d(
    const S* __restrict__ means3d,
    const int32_t* __restrict__ radii,
    int32_t* __restrict__ tiles_per_gauss,
    uint32_t N,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    if (N == 0) return;
    
    const uint32_t n_blocks = (N + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    compute_tiles_per_gauss_3d_kernel<S><<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        means3d,
        radii,
        tiles_per_gauss,
        N,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0
    );
}

template <typename S>
void launch_generate_isect_records_3d(
    const S* __restrict__ means3d,
    const int32_t* __restrict__ radii,
    const int64_t* __restrict__ cum_tiles_per_gauss,
    int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ flatten_ids,
    uint32_t N,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    if (N == 0) return;
    
    const uint32_t n_blocks = (N + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    generate_isect_records_3d_kernel<S><<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        means3d,
        radii,
        cum_tiles_per_gauss,
        isect_ids,
        flatten_ids,
        N,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0
    );
}

void launch_isect_offset_encode_3d(
    const int64_t* __restrict__ isect_ids,
    int32_t* __restrict__ offsets,
    uint32_t n_isects,
    uint32_t n_tiles,
    cudaStream_t stream
) {
    if (n_isects == 0) return;
    
    const uint32_t n_blocks = (n_isects + R2SPLAT_N_THREADS - 1) / R2SPLAT_N_THREADS;
    
    isect_offset_encode_3d_kernel<<<n_blocks, R2SPLAT_N_THREADS, 0, stream>>>(
        isect_ids,
        offsets,
        n_isects,
        n_tiles
    );
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

template void launch_compute_tiles_per_gauss_3d<float>(
    const float*, const int32_t*, int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_compute_tiles_per_gauss_3d<double>(
    const double*, const int32_t*, int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_generate_isect_records_3d<float>(
    const float*, const int32_t*, const int64_t*,
    int64_t*, int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

template void launch_generate_isect_records_3d<double>(
    const double*, const int32_t*, const int64_t*,
    int64_t*, int32_t*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t
);

// ============================================================================
// Radix Sort for Intersection IDs (using CUB)
// ============================================================================

std::pair<bool, bool> launch_radix_sort_pairs_3d(
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
    return std::make_pair(d_keys.selector == 0, d_values.selector == 0);
}

}  // namespace r2splat
