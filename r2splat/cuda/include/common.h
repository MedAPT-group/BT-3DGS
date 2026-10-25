/**
 * Common macros and type definitions for CUDA extensions
 * 
 * Following gsplat's pattern with templates, __restrict__, and common macros.
 * 
 * Author: Dy164
 * Date: 2025-12-17
 */

#ifndef R2SPLAT_CUDA_COMMON_H
#define R2SPLAT_CUDA_COMMON_H

#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

// ============================================================================
// Common Macros (following gsplat style)
// ============================================================================

#define R2SPLAT_N_THREADS 256

#define R2SPLAT_CHECK_CUDA(x)                                                \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define R2SPLAT_CHECK_CONTIGUOUS(x)                                          \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define R2SPLAT_CHECK_INPUT(x)                                               \
    R2SPLAT_CHECK_CUDA(x);                                                   \
    R2SPLAT_CHECK_CONTIGUOUS(x)
#define R2SPLAT_DEVICE_GUARD(_ten)                                           \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define R2SPLAT_PRAGMA_UNROLL _Pragma("unroll")

// ============================================================================
// Type Definitions
// ============================================================================

namespace r2splat {

// Operation type mapping for mixed precision support
template <typename T> struct OpType {
    typedef T type;
};

#ifdef __CUDA_ARCH__
template <> struct OpType<__nv_bfloat16> {
    typedef float type;
};

template <> struct OpType<__half> {
    typedef float type;
};
#endif

template <> struct OpType<c10::Half> {
    typedef float type;
};

template <> struct OpType<c10::BFloat16> {
    typedef float type;
};

} // namespace r2splat

#endif // R2SPLAT_CUDA_COMMON_H
