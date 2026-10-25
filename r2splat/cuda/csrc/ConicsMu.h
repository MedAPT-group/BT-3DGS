/**
 * CUDA kernel launch declarations for conic and compensation-factor transform.
 */

#ifndef R2SPLAT_CONICS_MU_H
#define R2SPLAT_CONICS_MU_H

#include "../include/common.h"
#include <cuda_runtime.h>

namespace r2splat {

template <typename S>
void launch_conics_mu_fwd(
    const S* __restrict__ cov2d_3x3,
    S eps,
    S* __restrict__ conics,
    S* __restrict__ mus,
    uint32_t total,
    cudaStream_t stream);

template <typename S>
void launch_conics_mu_bwd(
    const S* __restrict__ cov2d_3x3,
    S eps,
    const S* __restrict__ conics,
    const S* __restrict__ mus,
    const S* __restrict__ v_conics,
    const S* __restrict__ v_mus,
    S* __restrict__ v_cov2d_3x3,
    uint32_t total,
    cudaStream_t stream);

} // namespace r2splat

#endif // R2SPLAT_CONICS_MU_H
