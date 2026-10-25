/*NOTES by Dy164:
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
Keep the following correspondance in mind:
    - px -- k -- dim2 -- x (fastest varying, columns)
    - py -- j -- dim1 -- y (middle dimension)
    - pz -- i -- dim0 -- z (slowest varying, rows)
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
*/
/**
 * VoxelizeToVoxels Forward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based voxelization
 * specifically designed for CT reconstruction.
 * 
 * Key features:
 * - Single channel output (grayscale for CT)
 * - No color array support
 * - No mask support
 * - Tile-based with shared memory optimization
 * - CT formula: voxel_value = Σ (opacity × exp(-sigma))
 * 
 * Adapted from original gsplat VoxelizeToVoxelsFwd for r2splat.
 * 
 * Author: Dy164
 * Date: 2026-02-02
 */

#include "VoxelizeToVoxels.h"
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
 * Voxelize Gaussians to voxels for CT reconstruction.
 * 
 * Each block handles one tile (tile_size × tile_size × tile_size voxels).
 * Threads cooperatively load Gaussian data into shared memory,
 * then each thread accumulates contributions to its voxel.
 * 
 * CT formula per voxel:
 *   voxel_value = Σ_i (opacity_i × exp(-sigma_i))
 *   where sigma_i = 0.5 × delta^T × conic × delta
 *         delta = (px - mean_x, py - mean_y, pz - mean_z)
 * 
 * Conic is the inverse covariance matrix stored as upper triangular:
 *   [c00, c01, c02]
 *   [c01, c11, c12]  stored as [c00, c01, c02, c11, c12, c22]
 *   [c02, c12, c22]
 * 
 * Dimension convention:
 *   - block.group_index().x -> dim0 (z, slowest)
 *   - block.group_index().y -> dim1 (y, middle)
 *   - block.group_index().z -> dim2 (x, fastest)
 */
template <typename S>
__global__ void voxelize_to_voxels_fwd_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const vec3<S>* __restrict__ means3d,    // [N, 3] as (x, y, z)
    const S* __restrict__ conics,           // [N, 6] as [c00, c01, c02, c11, c12, c22]
    const S* __restrict__ opacities,        // [N]
    const uint32_t dim2,                    // x size (fastest)
    const uint32_t dim1,                    // y size (middle)
    const uint32_t dim0,                    // z size (slowest)
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,            // x tiles (fastest)
    const uint32_t n_tiles_dim1,            // y tiles (middle)
    const uint32_t n_tiles_dim0,            // z tiles (slowest)
    const int32_t* __restrict__ tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    S* __restrict__ render_output           // [dim0, dim1, dim2]
) {
    // Each thread draws one voxel, but also timeshares caching Gaussians
    // in a shared tile

    auto block = cg::this_thread_block();
    
    // Block indices: x=dim0(z), y=dim1(y), z=dim2(x)
    // tile_id = dim0 * (n_tiles_dim1 * n_tiles_dim2) + dim1 * n_tiles_dim2 + dim2
    int32_t tile_id = block.group_index().x * n_tiles_dim1 * n_tiles_dim2 +
                      block.group_index().y * n_tiles_dim2 +
                      block.group_index().z;
    
    // Voxel indices: i=dim0(z), j=dim1(y), k=dim2(x)
    uint32_t i = block.group_index().x * tile_size + block.thread_index().x;  // dim0, z
    uint32_t j = block.group_index().y * tile_size + block.thread_index().y;  // dim1, y
    uint32_t k = block.group_index().z * tile_size + block.thread_index().z;  // dim2, x

    // Voxel center coordinates (px=x, py=y, pz=z)
    S px = static_cast<S>(k) + static_cast<S>(0.5);  // x = dim2
    S py = static_cast<S>(j) + static_cast<S>(0.5);  // y = dim1
    S pz = static_cast<S>(i) + static_cast<S>(0.5);  // z = dim0
    
    // Voxel linear index: i * (dim1 * dim2) + j * dim2 + k
    int32_t vox_id = i * dim1 * dim2 + j * dim2 + k;

    // Return if out of bounds
    // Keep not rasterizing threads around for reading data
    bool inside = (i < dim0 && j < dim1 && k < dim2);
    bool done = !inside;

    // Have all threads in tile process the same Gaussians in batches
    // First collect Gaussians between range.x and range.y in batches
    // Which Gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == static_cast<int32_t>(n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2) - 1)
                            ? static_cast<int32_t>(n_isects)
                            : tile_offsets[tile_id + 1];
    
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout:
    // - id_batch: [block_size] Gaussian indices
    // - xyz_batch: [block_size] vec3 {x, y, z} centers
    // - conic_batch: [block_size * 6] conic elements
    // - opac_batch: [block_size] opacity values
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xyz_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    S* conic_batch = reinterpret_cast<S*>(&xyz_batch[block_size]);
    S* opac_batch = &conic_batch[block_size * 6];

    // Collect and process batches of Gaussians
    // Each thread loads one Gaussian at a time before rasterizing its
    // designated voxel
    uint32_t tr = block.thread_rank();

    S vox_out = static_cast<S>(0);

    for (uint32_t b = 0; b < num_batches; ++b) {
        // Resync all threads before beginning next batch
        // End early if entire tile is done
        if (__syncthreads_count(done) >= static_cast<int>(block_size)) {
            break;
        }

        // Each thread fetches 1 Gaussian from front to back
        // Index of Gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];  // Gaussian index
            id_batch[tr] = g;
            xyz_batch[tr] = means3d[g];
            opac_batch[tr] = opacities[g];
            // Load conic elements
            const S* conic_ptr = conics + g * 6;
            S* dest_ptr = conic_batch + tr * 6;
            dest_ptr[0] = conic_ptr[0];  // c00
            dest_ptr[1] = conic_ptr[1];  // c01
            dest_ptr[2] = conic_ptr[2];  // c02
            dest_ptr[3] = conic_ptr[3];  // c11
            dest_ptr[4] = conic_ptr[4];  // c12
            dest_ptr[5] = conic_ptr[5];  // c22
        }

        // Wait for other threads to collect the Gaussians in batch
        block.sync();

        // Process Gaussians in the current batch for this voxel
        uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> xyz = xyz_batch[t];
            const S opac = opac_batch[t];
            const S* conic_ptr = conic_batch + t * 6;
            
            // Compute delta = (px - mean_x, py - mean_y, pz - mean_z)
            const vec3<S> delta = {px - xyz.x, py - xyz.y, pz - xyz.z};
            
            // Compute sigma = 0.5 × delta^T × conic × delta
            // conic = [[c00, c01, c02], [c01, c11, c12], [c02, c12, c22]]
            // sigma = 0.5 * (c00*dx*dx + c11*dy*dy + c22*dz*dz)
            //       + c01*dx*dy + c02*dx*dz + c12*dy*dz
            const S c00 = conic_ptr[0];
            const S c01 = conic_ptr[1];
            const S c02 = conic_ptr[2];
            const S c11 = conic_ptr[3];
            const S c12 = conic_ptr[4];
            const S c22 = conic_ptr[5];
            
            const S sigma = static_cast<S>(0.5) * (
                c00 * delta.x * delta.x +
                c11 * delta.y * delta.y +
                c22 * delta.z * delta.z
            ) + c01 * delta.x * delta.y +
                c02 * delta.x * delta.z +
                c12 * delta.y * delta.z;

            // Compute alpha = opacity * exp(-sigma)
            S alpha = opac * __expf(-sigma);

            // Numerical stability check
            // Skip if sigma is negative (invalid conic) or alpha is too small
            if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-6)) {
                continue;
            }

            // Accumulate contribution
            vox_out += alpha;
        }
    }

    // Write output
    if (inside) {
        render_output[vox_id] = vox_out;
    }
}

// ============================================================================
// Launch Function
// ============================================================================

template <typename S>
void launch_voxelize_to_voxels_fwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    S* __restrict__ render_output,
    uint32_t N,
    uint32_t n_isects,
    uint32_t dim2,
    uint32_t dim1,
    uint32_t dim0,
    uint32_t tile_size,
    uint32_t n_tiles_dim2,
    uint32_t n_tiles_dim1,
    uint32_t n_tiles_dim0,
    cudaStream_t stream
) {
    // Each block covers a tile on the volume
    // In total there are n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2 blocks
    dim3 threads = {tile_size, tile_size, tile_size};
    dim3 blocks = {n_tiles_dim0, n_tiles_dim1, n_tiles_dim2};

    // Calculate shared memory size
    // id_batch: block_size * sizeof(int32_t)
    // xyz_batch: block_size * sizeof(vec3<S>)
    // conic_batch: block_size * 6 * sizeof(S)
    // opac_batch: block_size * sizeof(S)
    const uint32_t block_size = tile_size * tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +           // id_batch
        sizeof(vec3<S>) +           // xyz_batch
        6 * sizeof(S) +             // conic_batch
        sizeof(S)                   // opac_batch
    );

    // Set maximum shared memory size
    if (cudaFuncSetAttribute(
            voxelize_to_voxels_fwd_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        // Fallback if setting fails - kernel may still work with default shared memory
    }

    voxelize_to_voxels_fwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
        N,
        n_isects,
        reinterpret_cast<const vec3<S>*>(means3d),
        conics,
        opacities,
        dim2,
        dim1,
        dim0,
        tile_size,
        n_tiles_dim2,
        n_tiles_dim1,
        n_tiles_dim0,
        tile_offsets,
        flatten_ids,
        render_output
    );
}

template void launch_voxelize_to_voxels_fwd<float>(
    const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_fwd<double>(
    const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);


} // namespace r2splat

