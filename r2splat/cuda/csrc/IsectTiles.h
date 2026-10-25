/**
 * Isect Tiles CUDA Kernel Launch Functions
 * 
 * This header declares the launch functions that bridge between
 * C++ tensor processing (IsectTiles.cpp) and CUDA kernels (IsectTilesKernel.cu).
 * 
 * The module computes tile-Gaussian intersections for CT reconstruction.
 * Unlike alpha-blending rendering, CT reconstruction uses simple ray summation,
 * so depth ordering within tiles is not required.
 * 
 * Key simplifications from original gsplat:
 * - No packed mode support: input is always [C, N, ...]
 * - No depth sorting: only (camera_id, tile_id) ordering
 * - Split into two specialized kernels for better GPU efficiency
 * 
 * Author: Dy164
 * Date: 2025-01-12
 */

#ifndef R2SPLAT_ISECT_TILES_H
#define R2SPLAT_ISECT_TILES_H

#include "../include/common.h"
#include <cuda_runtime.h>
#include <utility>

namespace r2splat {

// ============================================================================
// First Pass: Compute tiles per Gaussian
// ============================================================================

/**
 * Launch kernel to compute the number of tiles each Gaussian covers.
 * 
 * This is the first pass of the two-pass algorithm.
 * After this, use cumsum to get write offsets for each Gaussian.
 * 
 * Dimension convention (following RasterizeToPixelsFwd.cu):
 *   - dim1 = x = columns (tile_id varies fastest)
 *   - dim0 = y = rows
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means2d 2D projected centers [C * N, 2], format: (x, y) = (dim1, dim0)
 * @param radii Gaussian radii in pixels [C * N] or rectangular radii [C * N, 2]
 * @param radii_is_rect Whether radii stores (rx, ry) pairs instead of a scalar
 * @param tiles_per_gauss Output: number of tiles per Gaussian [C * N]
 * @param C Number of cameras
 * @param N Number of Gaussians per camera
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in x dimension (columns)
 * @param n_tiles_dim0 Number of tiles in y dimension (rows)
 * @param stream CUDA stream
 */
template <typename S>
void launch_compute_tiles_per_gauss(
    const S* __restrict__ means2d,          // [C * N, 2]
    const int32_t* __restrict__ radii,      // [C * N] or [C * N, 2]
    int32_t* __restrict__ tiles_per_gauss,  // [C * N]
    uint32_t C,
    uint32_t N,
    bool radii_is_rect,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
);

// ============================================================================
// Second Pass: Generate intersection records
// ============================================================================

/**
 * Launch kernel to generate complete intersection records.
 * 
 * This is the second pass of the two-pass algorithm.
 * Requires cumsum of tiles_per_gauss to determine write offsets.
 * 
 * Dimension convention:
 *   - dim1 = x = columns (tile_id varies fastest)
 *   - dim0 = y = rows
 *   - tile_id = dim0 * n_tiles_dim1 + dim1
 * 
 * @tparam S Scalar type (float, double)
 * 
 * @param means2d 2D projected centers [C * N, 2]
 * @param radii Gaussian radii in pixels [C * N] or rectangular radii [C * N, 2]
 * @param radii_is_rect Whether radii stores (rx, ry) pairs instead of a scalar
 * @param cum_tiles_per_gauss Cumulative sum of tiles_per_gauss [C * N]
 * @param isect_ids Output: encoded (camera_id, tile_id) for sorting [n_isects]
 * @param flatten_ids Output: original Gaussian index [n_isects]
 * @param C Number of cameras
 * @param N Number of Gaussians per camera
 * @param tile_size Size of each tile in pixels
 * @param n_tiles_dim1 Number of tiles in x dimension (columns)
 * @param n_tiles_dim0 Number of tiles in y dimension (rows)
 * @param n_tiles Total number of tiles (n_tiles_dim0 * n_tiles_dim1)
 * @param stream CUDA stream
 */
template <typename S>
void launch_generate_isect_records(
    const S* __restrict__ means2d,                   // [C * N, 2]
    const int32_t* __restrict__ radii,               // [C * N] or [C * N, 2]
    const int64_t* __restrict__ cum_tiles_per_gauss, // [C * N]
    int64_t* __restrict__ isect_ids,                 // [n_isects]
    int32_t* __restrict__ flatten_ids,               // [n_isects]
    uint32_t C,
    uint32_t N,
    bool radii_is_rect,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    uint32_t n_tiles,
    cudaStream_t stream
);

// ============================================================================
// Offset Encoding
// ============================================================================

/**
 * Launch kernel to encode tile offsets from sorted isect_ids.
 * 
 * Creates an offset array where offsets[c, dim0, dim1] gives the starting
 * index in the sorted flatten_ids for tile (c, dim0, dim1).
 * 
 * @param isect_ids Sorted encoded (camera_id, tile_id) [n_isects]
 * @param offsets Output: per-tile starting offsets [C, n_tiles_dim0, n_tiles_dim1]
 * @param n_isects Total number of intersections
 * @param C Number of cameras
 * @param n_tiles Total number of tiles per camera
 * @param tile_n_bits Number of bits used to encode tile_id (unused, for API compat)
 * @param stream CUDA stream
 */
void launch_isect_offset_encode(
    const int64_t* __restrict__ isect_ids,  // [n_isects]
    int32_t* __restrict__ offsets,          // [C, n_tiles_dim0, n_tiles_dim1]
    uint32_t n_isects,
    uint32_t C,
    uint32_t n_tiles,
    uint32_t tile_n_bits,
    cudaStream_t stream
);

// ============================================================================
// Radix Sort for Intersection IDs
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
 * @return Pair of booleans: (keys_swapped, values_swapped)
 *         If true, the sorted result is in the *_in buffer, not *_out
 */
std::pair<bool, bool> launch_radix_sort_pairs(
    int64_t* isect_ids_in,
    int64_t* isect_ids_out,
    int32_t* flatten_ids_in,
    int32_t* flatten_ids_out,
    int64_t n_isects,
    uint32_t end_bit,
    cudaStream_t stream
);

}  // namespace r2splat

#endif  // R2SPLAT_ISECT_TILES_H
