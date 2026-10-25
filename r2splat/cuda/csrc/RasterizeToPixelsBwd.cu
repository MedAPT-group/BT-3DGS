/*NOTES by Dy164:
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
If test this backward module, errors may arise, especially in conics gradients,
this is firstly because the amplitude of conics is large,
and therefore the absolute error in conics gradients may be large too;
the second reason is that the intersection records are generated using 3 sigma cutoff,
as a consequence of which, cutoff effects at tile boundaries may lead to inaccurate gradients.
To mitigate the cutoff effects, we can increase the cutoff to 4 sigma,
yet this will also increase the memory consumption and computation time.
===========================================
IMPORTANT! IMPORTANT! IMPORTANT!
===========================================
*/
/**
 * RasterizeToPixels Backward CUDA Kernels
 * 
 * This file implements CUDA kernels for tile-based rasterization backward pass
 * specifically designed for CT reconstruction.
 * 
 * Key differences from standard 3DGS rasterization backward:
 * - No alpha blending: CT uses simple summation, no cumulative transmittance
 * - Order-independent: No need to process Gaussians back-to-front
 * - Single channel output: No COLOR_DIM template parameter
 * - No masks support
 * 
 * CT forward formula: pixel_value = Σ (opacity × mu × exp(-sigma))
 * 
 * Gradients computed:
 * - v_means2d: gradient w.r.t. 2D centers
 * - v_conics: gradient w.r.t. inverse covariance matrices
 * - v_opacities: gradient w.r.t. opacities
 * - v_mus: gradient w.r.t. mu values
 * 
 * Author: Dy164
 * Date: 2025-01-27
 */

#include "RasterizeToPixels.h"
#include "../include/types.cuh"
#include "../include/helpers.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cmath>

namespace r2splat {

namespace cg = cooperative_groups;

// ============================================================================
// Backward Pass Kernel
// ============================================================================

/**
 * Backward pass for CT rasterization.
 * 
 * Each block handles one tile (tile_size × tile_size pixels).
 * Threads cooperatively load Gaussian data and accumulate gradients.
 * 
 * For CT reconstruction (simple summation, no alpha blending):
 *   O = Σ_i (opacity_i × mu_i × exp(-sigma_i))
 *   
 * Given v_render = ∂L/∂O from upstream, we compute:
 *   v_opacity_i = v_render × mu_i × exp(-sigma_i)
 *   v_mu_i = v_render × opacity_i × exp(-sigma_i)
 *   v_sigma_i = -v_render × opacity_i × mu_i × exp(-sigma_i) = -v_render × alpha_i
 *   
 * Then expand sigma = 0.5 × (a×δx² + c×δy²) + b×δx×δy where δ = (px - mean_x, py - mean_y):
 *   v_mean_x = v_sigma × (a×δx + b×δy) (note: ∂sigma/∂mean_x = -∂sigma/∂δx × 1)
 *   v_mean_y = v_sigma × (c×δy + b×δx)
 *   v_conic_a = v_sigma × 0.5 × δx²
 *   v_conic_b = v_sigma × δx × δy
 *   v_conic_c = v_sigma × 0.5 × δy²
 */
template <typename S>
__global__ void rasterize_to_pixels_bwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    // Forward inputs
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
    // Upstream gradient
    const S* __restrict__ v_render_output,     // [C, dim0, dim1]
    // Output gradients
    vec2<S>* __restrict__ v_means2d_abs,       // [C * N, 2] optional (can be nullptr)
    vec2<S>* __restrict__ v_means2d,           // [C * N, 2]
    vec3<S>* __restrict__ v_conics,            // [C * N, 3]
    S* __restrict__ v_opacities,               // [C * N]
    S* __restrict__ v_mus                      // [C * N]
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * n_tiles_dim1 + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    // Offset pointers for this camera
    tile_offsets += camera_id * n_tiles_dim0 * n_tiles_dim1;
    v_render_output += camera_id * dim0 * dim1;

    // Pixel center coordinates
    const S px = static_cast<S>(j) + static_cast<S>(0.5);
    const S py = static_cast<S>(i) + static_cast<S>(0.5);
    const int32_t pix_id = i * dim1 + j;

    // Check if inside valid pixel range
    const bool inside = (i < dim0 && j < dim1);

    // Get upstream gradient for this pixel
    const S v_render = inside ? v_render_output[pix_id] : static_cast<S>(0);

    // Which Gaussians to look through in this tile
    const int32_t range_start = tile_offsets[tile_id];
    const int32_t range_end = 
        (camera_id == static_cast<int32_t>(C) - 1) && 
        (tile_id == static_cast<int32_t>(n_tiles_dim1 * n_tiles_dim0) - 1)
            ? static_cast<int32_t>(n_isects)
            : tile_offsets[tile_id + 1];

    const uint32_t block_size = block.size();
    const uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    // Shared memory layout (same as forward)
    extern __shared__ int s[];
    int32_t* id_batch = reinterpret_cast<int32_t*>(s);
    vec3<S>* xy_opacity_batch = reinterpret_cast<vec3<S>*>(&id_batch[block_size]);
    vec3<S>* conic_batch = reinterpret_cast<vec3<S>*>(&xy_opacity_batch[block_size]);
    S* mu_batch = reinterpret_cast<S*>(&conic_batch[block_size]);

    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // Process batches of Gaussians (front to back, order doesn't matter for CT)
    for (uint32_t b = 0; b < num_batches; ++b) {
        // Sync before loading next batch
        block.sync();

        // Each thread fetches 1 Gaussian
        const uint32_t batch_start = range_start + block_size * b;
        const uint32_t idx = batch_start + tr;
        if (idx < static_cast<uint32_t>(range_end)) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            mu_batch[tr] = mus[g];
        }

        // Wait for all threads to load
        block.sync();

        // Process Gaussians in the current batch
        const uint32_t batch_size = min(block_size, static_cast<uint32_t>(range_end) - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            // Initialize local gradients
            vec2<S> v_xy_local = {static_cast<S>(0), static_cast<S>(0)};
            vec2<S> v_xy_abs_local = {static_cast<S>(0), static_cast<S>(0)};
            vec3<S> v_conic_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
            S v_opacity_local = static_cast<S>(0);
            S v_mu_local = static_cast<S>(0);

            bool valid = inside;
            if (valid) {
                const vec3<S> conic = conic_batch[t];
                const vec3<S> xy_opac = xy_opacity_batch[t];
                const S opac = xy_opac.z;
                const S mu = mu_batch[t];
                const vec2<S> delta = {px - xy_opac.x, py - xy_opac.y};

                // Compute sigma = 0.5 * delta^T * conic * delta
                const S sigma = static_cast<S>(0.5) * (
                    conic.x * delta.x * delta.x +
                    conic.z * delta.y * delta.y
                ) + conic.y * delta.x * delta.y;

                // Compute vis = exp(-sigma) and alpha = opacity * mu * vis
                const S vis = __expf(-sigma);
                const S alpha = opac * mu * vis;

                // Skip if sigma is negative or alpha is too small (same as forward)
                if (sigma < static_cast<S>(0) || fabsf(alpha) < static_cast<S>(1e-5)) {
                    valid = false;
                }

                if (valid) {
                    // v_sigma = -v_render * alpha
                    const S v_sigma = -v_render * alpha;

                    // Gradients w.r.t. opacity and mu
                    v_opacity_local = v_render * mu * vis;
                    v_mu_local = v_render * opac * vis;

                    // Gradients w.r.t. means2d
                    // δ = pixel - mean, so ∂δ/∂mean = -1
                    // sigma = 0.5 * (a*δx² + c*δy²) + b*δx*δy
                    // ∂sigma/∂δx = a*δx + b*δy
                    // ∂sigma/∂δy = c*δy + b*δx
                    // ∂sigma/∂mean_x = ∂sigma/∂δx * ∂δx/∂mean_x = (a*δx + b*δy) * (-1)
                    // ∂sigma/∂mean_y = ∂sigma/∂δy * ∂δy/∂mean_y = (c*δy + b*δx) * (-1)
                    // v_mean_x = v_sigma * ∂sigma/∂mean_x = -v_sigma * (a*δx + b*δy)
                    // v_mean_y = v_sigma * ∂sigma/∂mean_y = -v_sigma * (c*δy + b*δx)
                    v_xy_local.x = -v_sigma * (conic.x * delta.x + conic.y * delta.y);
                    v_xy_local.y = -v_sigma * (conic.z * delta.y + conic.y * delta.x);
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {fabs(v_xy_local.x), fabs(v_xy_local.y)};
                    }

                    // Gradients w.r.t. conics
                    // ∂sigma/∂a = 0.5 * δx²
                    // ∂sigma/∂b = δx * δy
                    // ∂sigma/∂c = 0.5 * δy²
                    v_conic_local.x = v_sigma * static_cast<S>(0.5) * delta.x * delta.x;
                    v_conic_local.y = v_sigma * delta.x * delta.y;
                    v_conic_local.z = v_sigma * static_cast<S>(0.5) * delta.y * delta.y;
                }
            }

            // Warp-level reduction
            r2splat::warpSum(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                r2splat::warpSum(v_xy_abs_local, warp);
            }
            r2splat::warpSum(v_conic_local, warp);
            r2splat::warpSum(v_opacity_local, warp);
            r2splat::warpSum(v_mu_local, warp);

            // First thread in warp atomically adds to global memory
            if (warp.thread_rank() == 0) {
                const int32_t g = id_batch[t];
                
                S* v_xy_ptr = reinterpret_cast<S*>(v_means2d) + 2 * g;
                atomicAdd(v_xy_ptr, v_xy_local.x);
                atomicAdd(v_xy_ptr + 1, v_xy_local.y);

                // AbsGrad (optional): accumulate absolute projected-mean gradients
                if (v_means2d_abs != nullptr) {
                    S* v_xy_abs_ptr = reinterpret_cast<S*>(v_means2d_abs) + 2 * g;
                    atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                S* v_conic_ptr = reinterpret_cast<S*>(v_conics) + 3 * g;
                atomicAdd(v_conic_ptr, v_conic_local.x);
                atomicAdd(v_conic_ptr + 1, v_conic_local.y);
                atomicAdd(v_conic_ptr + 2, v_conic_local.z);

                atomicAdd(v_opacities + g, v_opacity_local);
                atomicAdd(v_mus + g, v_mu_local);
            }
        }
    }
}

// ============================================================================
// Launch Function
// ============================================================================

template <typename S>
void launch_rasterize_to_pixels_bwd(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int32_t* __restrict__ tile_offsets,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means2d_abs,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
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
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, n_tiles_dim0, n_tiles_dim1};

    // Calculate shared memory size (same as forward)
    const uint32_t block_size = tile_size * tile_size;
    const uint32_t shared_mem = block_size * (
        sizeof(int32_t) +
        sizeof(vec3<S>) +
        sizeof(vec3<S>) +
        sizeof(S)
    );

    // Set maximum shared memory size
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_bwd_kernel<S>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_bwd_kernel<S><<<blocks, threads, shared_mem, stream>>>(
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
        v_render_output,
        reinterpret_cast<vec2<S>*>(v_means2d_abs),
        reinterpret_cast<vec2<S>*>(v_means2d),
        reinterpret_cast<vec3<S>*>(v_conics),
        v_opacities,
        v_mus
    );
}

// ============================================================================
// Per-Gaussian-Tile Backward Pass Kernel
// ============================================================================

template <typename S>
__global__ void rasterize_to_pixels_bwd_per_gaussian_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const vec2<S>* __restrict__ means2d,
    const vec3<S>* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const uint32_t dim1,
    const uint32_t dim0,
    const uint32_t tile_size,
    const uint32_t n_tiles_dim1,
    const uint32_t n_tiles_dim0,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    vec2<S>* __restrict__ v_means2d_abs,
    vec2<S>* __restrict__ v_means2d,
    vec3<S>* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus
) {
    auto block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const uint32_t warps_per_block = block.size() / warp.size();
    const uint32_t isect_idx =
        block.group_index().x * warps_per_block + warp.meta_group_rank();
    if (isect_idx >= n_isects) {
        return;
    }

    const uint32_t n_tiles = n_tiles_dim0 * n_tiles_dim1;
    const int64_t encoded_id = isect_ids[isect_idx];
    const uint32_t camera_id = static_cast<uint32_t>(encoded_id / n_tiles);
    if (camera_id >= C) {
        return;
    }
    const uint32_t tile_id = static_cast<uint32_t>(encoded_id - static_cast<int64_t>(camera_id) * n_tiles);
    const uint32_t tile_y = tile_id / n_tiles_dim1;
    const uint32_t tile_x = tile_id - tile_y * n_tiles_dim1;
    const uint32_t origin_y = tile_y * tile_size;
    const uint32_t origin_x = tile_x * tile_size;

    const int32_t g = flatten_ids[isect_idx];
    if (g < 0 || g >= static_cast<int32_t>(C * N)) {
        return;
    }

    const vec2<S> xy = means2d[g];
    const vec3<S> conic = conics[g];
    const S opac = opacities[g];
    const S mu = mus[g];
    const S* __restrict__ v_render_camera =
        v_render_output + static_cast<uint64_t>(camera_id) * dim0 * dim1;

    vec2<S> v_xy_local = {static_cast<S>(0), static_cast<S>(0)};
    vec2<S> v_xy_abs_local = {static_cast<S>(0), static_cast<S>(0)};
    vec3<S> v_conic_local = {static_cast<S>(0), static_cast<S>(0), static_cast<S>(0)};
    S v_opacity_local = static_cast<S>(0);
    S v_mu_local = static_cast<S>(0);

    const uint32_t pixels_per_tile = tile_size * tile_size;
    for (uint32_t local_idx = warp.thread_rank(); local_idx < pixels_per_tile; local_idx += warp.size()) {
        const uint32_t local_y = local_idx / tile_size;
        const uint32_t local_x = local_idx - local_y * tile_size;
        const uint32_t i = origin_y + local_y;
        const uint32_t j = origin_x + local_x;
        if (i >= dim0 || j >= dim1) {
            continue;
        }

        const S px = static_cast<S>(j) + static_cast<S>(0.5);
        const S py = static_cast<S>(i) + static_cast<S>(0.5);
        const vec2<S> delta = {px - xy.x, py - xy.y};
        const S sigma = static_cast<S>(0.5) * (
            conic.x * delta.x * delta.x +
            conic.z * delta.y * delta.y
        ) + conic.y * delta.x * delta.y;
        const S vis = exp(-sigma);
        const S alpha = opac * mu * vis;
        if (sigma < static_cast<S>(0) || fabs(alpha) < static_cast<S>(1e-5)) {
            continue;
        }

        const S v_render = v_render_camera[i * dim1 + j];
        const S v_sigma = -v_render * alpha;
        const vec2<S> v_xy_pix = {
            -v_sigma * (conic.x * delta.x + conic.y * delta.y),
            -v_sigma * (conic.z * delta.y + conic.y * delta.x)
        };

        v_xy_local.x += v_xy_pix.x;
        v_xy_local.y += v_xy_pix.y;
        if (v_means2d_abs != nullptr) {
            v_xy_abs_local.x += fabs(v_xy_pix.x);
            v_xy_abs_local.y += fabs(v_xy_pix.y);
        }
        v_conic_local.x += v_sigma * static_cast<S>(0.5) * delta.x * delta.x;
        v_conic_local.y += v_sigma * delta.x * delta.y;
        v_conic_local.z += v_sigma * static_cast<S>(0.5) * delta.y * delta.y;
        v_opacity_local += v_render * mu * vis;
        v_mu_local += v_render * opac * vis;
    }

    r2splat::warpSum(v_xy_local, warp);
    if (v_means2d_abs != nullptr) {
        r2splat::warpSum(v_xy_abs_local, warp);
    }
    r2splat::warpSum(v_conic_local, warp);
    r2splat::warpSum(v_opacity_local, warp);
    r2splat::warpSum(v_mu_local, warp);

    if (warp.thread_rank() == 0) {
        S* v_xy_ptr = reinterpret_cast<S*>(v_means2d) + 2 * g;
        atomicAdd(v_xy_ptr, v_xy_local.x);
        atomicAdd(v_xy_ptr + 1, v_xy_local.y);

        if (v_means2d_abs != nullptr) {
            S* v_xy_abs_ptr = reinterpret_cast<S*>(v_means2d_abs) + 2 * g;
            atomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
            atomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
        }

        S* v_conic_ptr = reinterpret_cast<S*>(v_conics) + 3 * g;
        atomicAdd(v_conic_ptr, v_conic_local.x);
        atomicAdd(v_conic_ptr + 1, v_conic_local.y);
        atomicAdd(v_conic_ptr + 2, v_conic_local.z);
        atomicAdd(v_opacities + g, v_opacity_local);
        atomicAdd(v_mus + g, v_mu_local);
    }
}

template <typename S>
void launch_rasterize_to_pixels_bwd_per_gaussian(
    const S* __restrict__ means2d,
    const S* __restrict__ conics,
    const S* __restrict__ opacities,
    const S* __restrict__ mus,
    const int64_t* __restrict__ isect_ids,
    const int32_t* __restrict__ flatten_ids,
    const S* __restrict__ v_render_output,
    S* __restrict__ v_means2d_abs,
    S* __restrict__ v_means2d,
    S* __restrict__ v_conics,
    S* __restrict__ v_opacities,
    S* __restrict__ v_mus,
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
    constexpr uint32_t threads_per_block = 256;
    constexpr uint32_t warps_per_block = threads_per_block / 32;
    const uint32_t blocks = (n_isects + warps_per_block - 1) / warps_per_block;
    if (blocks == 0) {
        return;
    }

    rasterize_to_pixels_bwd_per_gaussian_kernel<S><<<blocks, threads_per_block, 0, stream>>>(
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
        isect_ids,
        flatten_ids,
        v_render_output,
        reinterpret_cast<vec2<S>*>(v_means2d_abs),
        reinterpret_cast<vec2<S>*>(v_means2d),
        reinterpret_cast<vec3<S>*>(v_conics),
        v_opacities,
        v_mus
    );
}

template void launch_rasterize_to_pixels_bwd<float>(
    const float*, const float*, const float*, const float*,
    const int32_t*, const int32_t*,
    const float*,
    float*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_bwd<double>(
    const double*, const double*, const double*, const double*,
    const int32_t*, const int32_t*,
    const double*,
    double*, double*, double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_bwd_per_gaussian<float>(
    const float*, const float*, const float*, const float*,
    const int64_t*, const int32_t*,
    const float*,
    float*, float*, float*, float*, float*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);

template void launch_rasterize_to_pixels_bwd_per_gaussian<double>(
    const double*, const double*, const double*, const double*,
    const int64_t*, const int32_t*,
    const double*,
    double*, double*, double*, double*, double*,
    uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
    cudaStream_t
);


} // namespace r2splat

