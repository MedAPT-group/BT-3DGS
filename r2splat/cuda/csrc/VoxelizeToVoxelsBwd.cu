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
 * VoxelizeToVoxels Backward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based voxelization backward pass
 * specifically designed for CT reconstruction.
 * 
 * Key features:
 * - Single channel output (grayscale for CT)
 * - No alpha blending: CT uses simple summation, order-independent
 * - Computes first-order gradients for voxelization
 * - Tile-based with shared memory optimization
 * 
 * CT forward formula: voxel_value = Σ (opacity × exp(-sigma))
 * where sigma = 0.5 × delta^T × conic × delta
 * 
 * Gradient notation:
 * - v_* : first-order gradient (∂L/∂*)
 * 
 * Conic storage: [N, 6] = [a, p, q, b, r, c] (upper triangular of 3x3 matrix)
 *   [[a, p, q],
 *    [p, b, r],
 *    [q, r, c]]
 * 
 * Author: Dy164
 * Date: 2026-02-03
 */

#include "VoxelizeToVoxels.h"
#include "../include/types.cuh"
#include "../include/helpers.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cmath>

namespace r2splat {

namespace cg = cooperative_groups;

// ============================================================================
// First-Order Backward Pass Kernel
// ============================================================================

/**
 * First-order backward pass for CT voxelization.
 * 
 * For CT reconstruction (simple summation, no alpha blending):
 *   O = Σ_g G_g where G = o × exp(-σ)
 *   
 * Given v_render = ∂L/∂O from upstream, we compute:
 *   v_opacity = v_O × exp(-σ)
 *   v_sigma = -v_O × o × exp(-σ) = -v_O × G
 *   
 * Then expand sigma = 0.5×(a×d1² + b×d2² + c×d3²) + p×d1×d2 + q×d1×d3 + r×d2×d3:
 *   v_mean = v_sigma × [a×d1+p×d2+q×d3, b×d2+p×d1+r×d3, c×d3+q×d1+r×d2]
 *   v_conic = v_sigma × [0.5×d1², d1×d2, d1×d3, 0.5×d2², d2×d3, 0.5×d3²]
 *           = v_sigma × [c00, c01, c02, c11, c12, c22] (storage order)
 */
template <typename S>
__global__ void voxelize_to_voxels_bwd_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
    const vec3<S>* __restrict__ means3d,    // [N, 3]
    const S* __restrict__ conics,           // [N, 6]
    const S* __restrict__ opacities,        // [N]
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int32_t* __restrict__ tile_offsets,  // [n_tiles_dim0, n_tiles_dim1, n_tiles_dim2]
    const int32_t* __restrict__ flatten_ids,   // [n_isects]
    // Upstream gradient
    const S* __restrict__ v_render_output,     // [dim0, dim1, dim2]
    // Output gradients
    vec3<S>* __restrict__ v_means3d,           // [N, 3]
    S* __restrict__ v_conics,                  // [N, 6]
    S* __restrict__ v_opacities                // [N]
) {
    auto block = cg::this_thread_block();
    
    int32_t tile_id = block.group_index().x * n_tiles_dim1 * n_tiles_dim2 +
                      block.group_index().y * n_tiles_dim2 +
                      block.group_index().z;
    
    uint32_t i = block.group_index().x * tile_size + block.thread_index().x;  // dim0
    uint32_t j = block.group_index().y * tile_size + block.thread_index().y;  // dim1
    uint32_t k = block.group_index().z * tile_size + block.thread_index().z;  // dim2

    // Voxel center coordinates
    const S px = static_cast<S>(k) + static_cast<S>(0.5);
    const S py = static_cast<S>(j) + static_cast<S>(0.5);
    const S pz = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t vox_id = i * dim1 * dim2 + j * dim2 + k;

    const bool inside = (i < dim0 && j < dim1 && k < dim2);

    // Get upstream gradient
    const S v_O = inside ? v_render_output[vox_id] : static_cast<S>(0);

    // Tile range
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = (tile_id == static_cast<int32_t>(n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2) - 1)
                                  ? static_cast<int32_t>(n_isects)
                                  : tile_offsets[tile_id + 1];
    
    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xyz_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    S* conic_batch = reinterpret_cast<S*>(&xyz_batch[block_size]);
    S* opac_batch = &conic_batch[block_size * 6];

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    for (uint32_t b = 0; b < num_batches; ++b) {
        block.sync();

        // Load Gaussian data
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            xyz_batch[tr] = means3d[g];
            opac_batch[tr] = opacities[g];
            const S* conic_ptr = conics + g * 6;
            S* dest_ptr = conic_batch + tr * 6;
            #pragma unroll
            for (int c = 0; c < 6; ++c) {
                dest_ptr[c] = conic_ptr[c];
            }
        }
        block.sync();

        uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            bool valid = inside;
            S opac, vis;
            vec3<S> delta;
            S conic[6];

            if (valid) {
                vec3<S> xyz = xyz_batch[t];
                opac = opac_batch[t];
                const S* conic_ptr = conic_batch + t * 6;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    conic[c] = conic_ptr[c];
                }

                delta = {px - xyz.x, py - xyz.y, pz - xyz.z};
                
                // sigma = 0.5*(a*d1^2 + b*d2^2 + c*d3^2) + p*d1*d2 + q*d1*d3 + r*d2*d3
                // conic = [a, p, q, b, r, c]
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                const S d1 = delta.x, d2 = delta.y, d3 = delta.z;
                
                const S sigma = static_cast<S>(0.5) * (a*d1*d1 + bb*d2*d2 + c*d3*d3) +
                                p*d1*d2 + q*d1*d3 + r*d2*d3;
                vis = __expf(-sigma);
                const S alpha = opac * vis;
                
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-6)) {
                    valid = false;
                }
            }

            if (!warp.any(valid)) {
                continue;
            }

            // Initialize local gradients
            vec3<S> v_xyz_local = {0, 0, 0};
            S v_conic_local[6] = {0, 0, 0, 0, 0, 0};
            S v_opacity_local = 0;

            if (valid) {
                const S a = conic[0], p = conic[1], q = conic[2];
                const S bb = conic[3], r = conic[4], c = conic[5];
                const S d1 = delta.x, d2 = delta.y, d3 = delta.z;

                // v_sigma = -v_O * o * vis (used for conic gradients)
                const S v_sigma = -v_O * opac * vis;
                
                // oI = o * vis (used for mean gradients)
                const S oI = opac * vis;

                // v_opacity = v_O * vis
                v_opacity_local = v_O * vis;

                // v_conic = v_sigma * [0.5*d1^2, d1*d2, d1*d3, 0.5*d2^2, d2*d3, 0.5*d3^2]
                v_conic_local[0] = v_sigma * static_cast<S>(0.5) * d1 * d1;  // a
                v_conic_local[1] = v_sigma * d1 * d2;                        // p
                v_conic_local[2] = v_sigma * d1 * d3;                        // q
                v_conic_local[3] = v_sigma * static_cast<S>(0.5) * d2 * d2;  // b
                v_conic_local[4] = v_sigma * d2 * d3;                        // r
                v_conic_local[5] = v_sigma * static_cast<S>(0.5) * d3 * d3;  // c

                // v_mean = v_O * oI * s (where s = Σ^{-1} @ d)
                // Note: dG/dm = oI * s (positive sign, not v_sigma!)
                v_xyz_local.x = v_O * oI * (a*d1 + p*d2 + q*d3);
                v_xyz_local.y = v_O * oI * (bb*d2 + p*d1 + r*d3);
                v_xyz_local.z = v_O * oI * (c*d3 + q*d1 + r*d2);
            }

            // Warp reduction
            r2splat::warpSum(v_xyz_local, warp);
            r2splat::warpSum<6>(v_conic_local, warp);
            r2splat::warpSum(v_opacity_local, warp);

            // Atomic add to global memory (only warp leader)
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t];
                
                S* v_xyz_ptr = reinterpret_cast<S*>(v_means3d) + 3 * g;
                atomicAdd(v_xyz_ptr, v_xyz_local.x);
                atomicAdd(v_xyz_ptr + 1, v_xyz_local.y);
                atomicAdd(v_xyz_ptr + 2, v_xyz_local.z);

                S* v_conic_ptr = v_conics + 6 * g;
                #pragma unroll
                for (int c = 0; c < 6; ++c) {
                    atomicAdd(v_conic_ptr + c, v_conic_local[c]);
                }

                atomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

template <typename S>
void launch_voxelize_to_voxels_bwd(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
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
    dim3 threads = {tile_size, tile_size, tile_size};
    dim3 blocks = {n_tiles_dim0, n_tiles_dim1, n_tiles_dim2};

    const uint32_t block_size = tile_size * tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        6 * sizeof(S) +
        sizeof(S)
    );

    cudaFuncSetAttribute(
        voxelize_to_voxels_bwd_kernel<S>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem
    );

    voxelize_to_voxels_bwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
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
        v_render_output,
        reinterpret_cast<vec3<S>*>(v_means3d),
        v_conics,
        v_opacities
    );
}

// ============================================================================
// Per-Gaussian-Tile First-Order Backward Pass Kernel
// ============================================================================

template <typename S>
__global__ void voxelize_to_voxels_bwd_per_gaussian_kernel(
    const uint32_t N,
    const uint32_t n_isects,
    const vec3<S>* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const uint32_t dim2,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim2,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    vec3<S>* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t warps_per_block = block.size() / warp.size();
    const uint32_t isect_idx =
        block.group_index().x * warps_per_block + warp.meta_group_rank();
    if (isect_idx >= n_isects) {
        return;
    }

    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1 * n_tiles_dim2;
    const uint32_t tile_id = static_cast<uint32_t>(isect_ids[isect_idx]);
    if (tile_id >= n_tiles) {
        return;
    }

    const uint32_t tiles_per_z = n_tiles_dim1 * n_tiles_dim2;
    const uint32_t tile_dim0 = tile_id / tiles_per_z;
    const uint32_t rem0 = tile_id - tile_dim0 * tiles_per_z;
    const uint32_t tile_dim1 = rem0 / n_tiles_dim2;
    const uint32_t tile_dim2 = rem0 - tile_dim1 * n_tiles_dim2;
    const uint32_t origin_i = tile_dim0 * tile_size;
    const uint32_t origin_j = tile_dim1 * tile_size;
    const uint32_t origin_k = tile_dim2 * tile_size;

    const int32_t g = flatten_ids[isect_idx];
    if (g < 0 || g >= static_cast<int32_t>(N)) {
        return;
    }

    const vec3<S> xyz = means3d[g];
    const S opac = opacities[g];
    const S* conic_ptr = conics + g * 6;
    const S a = conic_ptr[0], p = conic_ptr[1], q = conic_ptr[2];
    const S bb = conic_ptr[3], r = conic_ptr[4], c = conic_ptr[5];

    vec3<S> v_xyz_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
    S v_conic_local[6] = {0, 0, 0, 0, 0, 0};
    S v_opacity_local = static_cast<S>(0);

    const uint32_t voxels_per_tile = tile_size * tile_size * tile_size;
    for (uint32_t local_idx = warp.thread_rank(); local_idx < voxels_per_tile; local_idx += warp.size()) {
        const uint32_t local_i = local_idx / (tile_size * tile_size);
        const uint32_t rem = local_idx - local_i * tile_size * tile_size;
        const uint32_t local_j = rem / tile_size;
        const uint32_t local_k = rem - local_j * tile_size;
        const uint32_t i = origin_i + local_i;
        const uint32_t j = origin_j + local_j;
        const uint32_t k = origin_k + local_k;
        if (i >= dim0 || j >= dim1 || k >= dim2) {
            continue;
        }

        const S px = static_cast<S>(k) + static_cast<S>(0.5);
        const S py = static_cast<S>(j) + static_cast<S>(0.5);
        const S pz = static_cast<S>(i) + static_cast<S>(0.5);
        const S d1 = px - xyz.x;
        const S d2 = py - xyz.y;
        const S d3 = pz - xyz.z;
        const S sigma = static_cast<S>(0.5) * (a*d1*d1 + bb*d2*d2 + c*d3*d3) +
                        p*d1*d2 + q*d1*d3 + r*d2*d3;
        const S vis = exp(-sigma);
        const S alpha = opac * vis;
        if (sigma < static_cast<S>(0) || fabs(alpha) < static_cast<S>(1e-6)) {
            continue;
        }

        const S v_O = v_render_output[i * dim1 * dim2 + j * dim2 + k];
        const S v_sigma = -v_O * opac * vis;
        const S oI = opac * vis;

        v_opacity_local += v_O * vis;
        v_conic_local[0] += v_sigma * static_cast<S>(0.5) * d1 * d1;
        v_conic_local[1] += v_sigma * d1 * d2;
        v_conic_local[2] += v_sigma * d1 * d3;
        v_conic_local[3] += v_sigma * static_cast<S>(0.5) * d2 * d2;
        v_conic_local[4] += v_sigma * d2 * d3;
        v_conic_local[5] += v_sigma * static_cast<S>(0.5) * d3 * d3;
        v_xyz_local.x += v_O * oI * (a*d1 + p*d2 + q*d3);
        v_xyz_local.y += v_O * oI * (bb*d2 + p*d1 + r*d3);
        v_xyz_local.z += v_O * oI * (c*d3 + q*d1 + r*d2);
    }

    r2splat::warpSum(v_xyz_local, warp);
    r2splat::warpSum<6>(v_conic_local, warp);
    r2splat::warpSum(v_opacity_local, warp);

    if (warp.thread_rank() == 0) {
        S* v_xyz_ptr = reinterpret_cast<S*>(v_means3d) + 3 * g;
        atomicAdd(v_xyz_ptr, v_xyz_local.x);
        atomicAdd(v_xyz_ptr + 1, v_xyz_local.y);
        atomicAdd(v_xyz_ptr + 2, v_xyz_local.z);

        S* v_conic_ptr = v_conics + 6 * g;
        #pragma unroll
        for (int idx = 0; idx < 6; ++idx) {
            atomicAdd(v_conic_ptr + idx, v_conic_local[idx]);
        }
        atomicAdd(v_opacities + g, v_opacity_local);
    }
}

template <typename S>
void launch_voxelize_to_voxels_bwd_per_gaussian(
    const S* __restrict__ means3d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means3d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
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
    constexpr uint32_t threads_per_block = 256;
    constexpr uint32_t warps_per_block = threads_per_block / 32;
    const uint32_t blocks = (n_isects + warps_per_block - 1) / warps_per_block;
    if (blocks == 0) {
        return;
    }

    voxelize_to_voxels_bwd_per_gaussian_kernel<S><<<blocks, threads_per_block, 0, stream>>>(
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
        isect_ids,
        flatten_ids,
        v_render_output,
        reinterpret_cast<vec3<S>*>(v_means3d),
        v_conics,
        v_opacities
    );
}

template void launch_voxelize_to_voxels_bwd<float>(
    const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*,
    float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd<double>(
    const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*,
    double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd_per_gaussian<float>(
    const float*, const float*, const float*,
    const int64_t*, const int32_t*,
    const float*,
    float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_voxelize_to_voxels_bwd_per_gaussian<double>(
    const double*, const double*, const double*,
    const int64_t*, const int32_t*,
    const double*,
    double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    cudaStream_t
);


} // namespace r2splat

