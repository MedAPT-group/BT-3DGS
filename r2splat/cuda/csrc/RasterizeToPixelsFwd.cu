/*NOTES by Dy164:
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
Keep the following correspondance in mind:
    - px -- j -- dim1 -- width(legacy)
    - py -- i -- dim0 -- height(legacy)
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
*/
/**
 * RasterizeToPixels Forward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based rasterization
 * specifically designed for CT reconstruction.
 * 
 * Key features:
 * - Single channel output (no color array)
 * - No packed mode support
 * - Tile-based with shared memory optimization
 * - CT formula: pixel_value = Σ (opacity × mu × exp(-sigma))
 * 
 * Following gsplat style for numerical stability and performance.
 * 
 * Author: Dy164
 * Date: 2025-01-22
 */

#include "RasterizeToPixels.h"
#include "../include/types.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cmath>

namespace r2splat {

namespace cg = cooperative_groups;

// ============================================================================
// Forward Pass Kernel
// ============================================================================

/**
 * Rasterize Gaussians to pixels for CT reconstruction.
 * 
 * Each block handles one tile (tile_size × tile_size pixels).
 * Threads cooperatively load Gaussian data into shared memory,
 * then each thread accumulates contributions to its pixel.
 * 
 * CT formula per pixel:
 *   pixel_value = Σ_i (opacity_i × mu_i × exp(-sigma_i))
 *   where sigma_i = 0.5 × delta^T × conic × delta
 *         delta = (px - mean_x, py - mean_y)
 */
template <typename S>
__global__ void rasterize_to_pixels_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<S>* __restrict__ means2d,    // [C * N, 2]
    const vec3<S>* __restrict__ conics,     // [C * N, 3]
    const S* __restrict__ opacities,        // [C * N]
    const S* __restrict__ mus,              // [C * N]
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,  // [C, n_tiles_dim0, n_tiles_dim1]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    S* __restrict__ render_output           // [C, dim0, dim1]
) {
    // Each thread draws one pixel, but also timeshares caching Gaussians
    // in a shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * n_tiles_dim1 + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Offset pointers for this camera
    tile_offsets += camera_id * n_tiles_dim0 * n_tiles_dim1;
    render_output += camera_id * dim0 * dim1;

    // Pixel center coordinates
    S px = static_cast<S>(j) + static_cast<S>(0.5);
    S py = static_cast<S>(i) + static_cast<S>(0.5);
    int32_t pix_id = i * dim1 + j;

    // Return if out of bounds
    // Keep not rasterizing threads around for reading data
    bool inside = (i < dim0 && j < dim1);
    bool done = !inside;

    // Have all threads in tile process the same Gaussians in batches
    // First collect Gaussians between range.x and range.y in batches
    // Which Gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = 
        (camera_id == static_cast<int32_t>(C) - 1) && 
        (tile_id == static_cast<int32_t>(n_tiles_dim1 * n_tiles_dim0) - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];
    
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout:
    // - id_batch: [block_size] Gaussian indices
    // - xy_opacity_batch: [block_size] vec3 {x, y, opacity}
    // - conic_batch: [block_size] vec3 {a, b, c}
    // - mu_batch: [block_size] mu values
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xy_opacity_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    vec3<S>* conic_batch = reinterpret_cast<vec3<S>*>(&xy_opacity_batch[block_size]);
    S* mu_batch = reinterpret_cast<S*>(&conic_batch[block_size]);

    // Collect and process batches of Gaussians
    // Each thread loads one Gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out = static_cast<S>(0);

    for (uint32_t b = 0; b < num_batches; ++b) {
        // Resync all threads before beginning next batch
        // End early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // Each thread fetches 1 Gaussian from front to back
        // Index of Gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];  // Flatten index in [C * N]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            mu_batch[tr] = mus[g];
        }

        // Wait for other threads to collect the Gaussians in batch
        block.sync();

        // Process Gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const S mu = mu_batch[t];
            const vec2<S> delta = {px - xy_opac.x, py - xy_opac.y};
            
            // Compute sigma = 0.5 * delta^T * conic * delta
            // conic = [a, b; b, c] stored as {a, b, c}
            const S sigma = static_cast<S>(0.5) * (
                conic.x * delta.x * delta.x +
                conic.z * delta.y * delta.y
            ) + conic.y * delta.x * delta.y;

            // Compute alpha = opacity * mu * exp(-sigma)
            S alpha = opac * mu * __expf(-sigma);

            // Numerical stability check (following gsplat style)
            // Skip if sigma is negative (invalid conic) or alpha is too small
            if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                continue;
            }

            // Accumulate contribution
            pix_out += alpha;
        }
    }

    // Write output
    if (inside) {
        render_output[pix_id] = pix_out;
    }
}

// ============================================================================
// Launch Function
// ============================================================================

template <typename S>
void launch_rasterize_to_pixels_fwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    uint32_t C,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    // Each block covers a tile on the image
    // In total there are C * n_tiles_dim0 * n_tiles_dim1 blocks
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, n_tiles_dim0, n_tiles_dim1};

    // Calculate shared memory size
    // id_batch: block_size * sizeof(int32_t)
    // xy_opacity_batch: block_size * sizeof(vec3<S>)
    // conic_batch: block_size * sizeof(vec3<S>)
    // mu_batch: block_size * sizeof(S)
    const uint32_t block_size = tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        sizeof(vec3<S>) +
        sizeof(S)
    );

    // Set maximum shared memory size
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_fwd_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_fwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        C,
        N,
        n_isects,
        reinterpret_cast<const vec2<S>*>(means2d),
        reinterpret_cast<const vec3<S>*>(conics),
        opacities,
        mus,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        render_output
    );
}

template void launch_rasterize_to_pixels_fwd<float>(
    const float*, const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_fwd<double>(
    const double*, const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);


} // namespace r2splat

