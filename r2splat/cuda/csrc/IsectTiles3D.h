/**
 * Isect Tiles 3D CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (IsectTiles3D.cpp) and CUDA kernels (IsectTiles3DKernel.cu).
 * 
 * The module computes tile-Gaussian intersections for 3D CT reconstruction.
 * Unlike alpha-blending rendering, CT reconstruction uses simple ray summation,
 * so depth ordering within tiles is not required.
 * 
 * Key simplifications from original gsplat:
 * - No depth sorting: only tile_id ordering
 * - Split into two specialized kernels for better GPU efficiency
 * 
 * Author: Dy164
 * Date: 2025-01-12
 * Updated: 2026-02-02 (Refactored to split kernel pattern, removed depth)
 */

#ifndef R2SPLAT_ISECT_TILES_3D_H
#define R2SPLAT_ISECT_TILES_3D_H

#include "../include/common.h"
#include <cuda_runtime.h>
#include <utility>

namespace r2splat {

// ============================================================================
// First Pass: Compute tiles per Gaussian (3D)
// ============================================================================

/**
 * Launch kernel to compute the number of tiles each Gaussian covers in 3D.
 * 
 * This is the first pass of the two-pass algorithm.
 * After this, use cumsum to get write offsets for each Gaussian.
 * 
 * Dimension convention (following VoxelizeToVoxelsFwd.cu):
 *   - dim2 = x (fastest varying in memory)
 *   - dim1 = y (middle dimension)
 *   - dim0 = z (slowest varying in memory)
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means3d 3D Gaussian centers [N, 3], stored as (x, y, z)
 * @param radii Gaussian radii in each dimension [N, 3], stored as (rx, ry, rz)
 * @param tiles_per_gauss Output: number of tiles per Gaussian [N]
 * @param N Number of Gaussians
 * @param tile_size Size of each tile in voxels
 * @param n_tiles_dim2 Number of tiles in x dimension (fastest)
 * @param n_tiles_dim1 Number of tiles in y dimension (middle)
 * @param n_tiles_dim0 Number of tiles in z dimension (slowest)
 * @param stream CUDA stream
 */
template <typename S>
void launch_compute_tiles_per_gauss_3d(
    const S* __restrict__ means3d,          // [N, 3]
    const int32_t* __restrict__ radii,      // [N, 3]
    int32_t* __restrict__ tiles_per_gauss,  // [N]
    uint32_t N,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Second Pass: Generate intersection records (3D)
// ============================================================================

/**
 * Launch kernel to generate complete intersection records for 3D.
 * 
 * This is the second pass of the two-pass algorithm.
 * Requires cumsum of tiles_per_gauss to determine write offsets.
 * 
 * Dimension convention:
 *   - dim2 = x (fastest varying in memory)
 *   - dim1 = y (middle dimension)
 *   - dim0 = z (slowest varying in memory)
 *   - tile_id = dim0 * (n_tiles_dim1 * n_tiles_dim2) + dim1 * n_tiles_dim2 + dim2
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means3d 3D Gaussian centers [N, 3]
 * @param radii Gaussian radii in each dimension [N, 3]
 * @param cum_tiles_per_gauss Cumulative sum of tiles_per_gauss [N]
 * @param isect_ids Output: tile_id for sorting [n_isects]
 * @param flatten_ids Output: original Gaussian index [n_isects]
 * @param N Number of Gaussians
 * @param tile_size Size of each tile in voxels
 * @param n_tiles_dim2 Number of tiles in x dimension (fastest)
 * @param n_tiles_dim1 Number of tiles in y dimension (middle)
 * @param n_tiles_dim0 Number of tiles in z dimension (slowest)
 * @param stream CUDA stream
 */
template <typename S>
void launch_generate_isect_records_3d(
    const S* __restrict__ means3d,                   // [N, 3]
    const int32_t* __restrict__ radii,               // [N, 3]
    const int64_t* __restrict__ cum_tiles_per_gauss, // [N]
    int64_t* __restrict__ isect_ids,                 // [n_isects]
    int32_t* __restrict__ flatten_ids,               // [n_isects]
    uint32_t N,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Offset Encoding (3D)
// ============================================================================

/**
 * Launch kernel to encode tile offsets from sorted isect_ids for 3D.
 * 
 * Creates an offset array where offsets[dim0, dim1, dim2] gives the starting
 * index in the sorted flatten_ids for that tile.
 * 
 * @param isect_ids Sorted tile_ids [n_isects]
 * @param offsets Output: per-tile starting offsets [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
 * @param n_isects Total number of intersections
 * @param n_tiles Total number of tiles (n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2)
 * @param stream CUDA stream
 */
void launch_isect_offset_encode_3d(
    const int64_t* __restrict__ isect_ids,  // [n_isects]
    int32_t* __restrict__ offsets,          // [n_tiles]
    uint32_t n_isects,
    uint32_t n_tiles,
    cudaStream_t stream
);

// ============================================================================
// Radix Sort for Intersection IDs (3D)
// ============================================================================

/**
 * Sort isect_ids and flatten_ids by isect_ids using CUB radix sort.
 * 
 * This function must be in .cu file because CUB requires CUDA compilation.
 * Uses DoubleBuffer for efficient in-place sorting.
 * 
 * @param isect_ids_in Input isect_ids [n_isects]
 * @param isect_ids_out Output sorted isect_ids [n_isects]
 * @param flatten_ids_in Input flatten_ids [n_isects]
 * @param flatten_ids_out Output sorted flatten_ids [n_isects]
 * @param n_isects Number of intersections
 * @param end_bit Number of bits to sort (determines sort range)
 * @param stream CUDA stream
 * 
 * @return Pair of booleans: (keys_in_place, values_in_place)
 *         If true, the sorted result is in the *_in buffer, not *_out
 */
std::pair<bool, bool> launch_radix_sort_pairs_3d(
    int64_t* isect_ids_in,
    int64_t* isect_ids_out,
    int32_t* flatten_ids_in,
    int32_t* flatten_ids_out,
    int64_t n_isects,
    uint32_t end_bit,
    cudaStream_t stream
);

}  // namespace r2splat

#endif  // R2SPLAT_ISECT_TILES_3D_H
