/**
 * @file types.cuh
 * @brief CUDA-compatible vector and matrix types for r2splat
 * 
 * Provides simple vector (vec2, vec3, vec4) and matrix (mat2, mat3, mat4) types
 * for use in CUDA kernels. These are compatible with both host and device code.
 * 
 * Following gsplat's pattern for type definitions.
 * 
 * Author: Dy164
 * Date: 2025-01-12
 */

#ifndef R2SPLAT_TYPES_CUH
#define R2SPLAT_TYPES_CUH

#include <cuda_runtime.h>

namespace r2splat {

// ============================================================================
// Vector Types
// ============================================================================

/**
 * 2D vector type
 */
template <typename T>
struct vec2 {
    T x, y;
    
    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(T x, T y) : x(x), y(y) {}
    __host__ __device__ explicit vec2(T v) : x(v), y(v) {}
    
    __host__ __device__ vec2<T> operator+(const vec2<T>& v) const {
        return vec2<T>(x + v.x, y + v.y);
    }
    __host__ __device__ vec2<T> operator-(const vec2<T>& v) const {
        return vec2<T>(x - v.x, y - v.y);
    }
    __host__ __device__ vec2<T> operator*(T s) const {
        return vec2<T>(x * s, y * s);
    }
    __host__ __device__ vec2<T> operator*(const vec2<T>& v) const {
        return vec2<T>(x * v.x, y * v.y);
    }
    __host__ __device__ vec2<T> operator/(T s) const {
        return vec2<T>(x / s, y / s);
    }
    __host__ __device__ vec2<T>& operator+=(const vec2<T>& v) {
        x += v.x; y += v.y;
        return *this;
    }
    __host__ __device__ vec2<T>& operator-=(const vec2<T>& v) {
        x -= v.x; y -= v.y;
        return *this;
    }
    __host__ __device__ vec2<T>& operator*=(T s) {
        x *= s; y *= s;
        return *this;
    }
};

/**
 * 3D vector type
 */
template <typename T>
struct vec3 {
    T x, y, z;
    
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(T x, T y, T z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit vec3(T v) : x(v), y(v), z(v) {}
    
    __host__ __device__ vec3<T> operator+(const vec3<T>& v) const {
        return vec3<T>(x + v.x, y + v.y, z + v.z);
    }
    __host__ __device__ vec3<T> operator-(const vec3<T>& v) const {
        return vec3<T>(x - v.x, y - v.y, z - v.z);
    }
    __host__ __device__ vec3<T> operator*(T s) const {
        return vec3<T>(x * s, y * s, z * s);
    }
    __host__ __device__ vec3<T> operator*(const vec3<T>& v) const {
        return vec3<T>(x * v.x, y * v.y, z * v.z);
    }
    __host__ __device__ vec3<T> operator/(T s) const {
        return vec3<T>(x / s, y / s, z / s);
    }
    __host__ __device__ vec3<T>& operator+=(const vec3<T>& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }
    __host__ __device__ vec3<T>& operator-=(const vec3<T>& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }
    __host__ __device__ vec3<T>& operator*=(T s) {
        x *= s; y *= s; z *= s;
        return *this;
    }
};

/**
 * 4D vector type
 */
template <typename T>
struct vec4 {
    T x, y, z, w;
    
    __host__ __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __host__ __device__ vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {}
    __host__ __device__ explicit vec4(T v) : x(v), y(v), z(v), w(v) {}
    
    __host__ __device__ vec4<T> operator+(const vec4<T>& v) const {
        return vec4<T>(x + v.x, y + v.y, z + v.z, w + v.w);
    }
    __host__ __device__ vec4<T> operator-(const vec4<T>& v) const {
        return vec4<T>(x - v.x, y - v.y, z - v.z, w - v.w);
    }
    __host__ __device__ vec4<T> operator*(T s) const {
        return vec4<T>(x * s, y * s, z * s, w * s);
    }
    __host__ __device__ vec4<T> operator*(const vec4<T>& v) const {
        return vec4<T>(x * v.x, y * v.y, z * v.z, w * v.w);
    }
    __host__ __device__ vec4<T>& operator+=(const vec4<T>& v) {
        x += v.x; y += v.y; z += v.z; w += v.w;
        return *this;
    }
};

// ============================================================================
// Matrix Types (column-major storage)
// ============================================================================

/**
 * 2x2 matrix type (column-major)
 */
template <typename T>
struct mat2 {
    vec2<T> cols[2];
    
    __host__ __device__ mat2() {}
    __host__ __device__ mat2(const vec2<T>& c0, const vec2<T>& c1) {
        cols[0] = c0; cols[1] = c1;
    }
    __host__ __device__ mat2(T m00, T m10, T m01, T m11) {
        cols[0] = vec2<T>(m00, m10);
        cols[1] = vec2<T>(m01, m11);
    }
    
    __host__ __device__ vec2<T>& operator[](int i) { return cols[i]; }
    __host__ __device__ const vec2<T>& operator[](int i) const { return cols[i]; }
    
    // Matrix-vector multiplication
    __host__ __device__ vec2<T> operator*(const vec2<T>& v) const {
        return vec2<T>(
            cols[0].x * v.x + cols[1].x * v.y,
            cols[0].y * v.x + cols[1].y * v.y
        );
    }
    
    // Matrix-matrix multiplication
    __host__ __device__ mat2<T> operator*(const mat2<T>& m) const {
        return mat2<T>(
            (*this) * m.cols[0],
            (*this) * m.cols[1]
        );
    }
};

/**
 * 3x3 matrix type (column-major)
 */
template <typename T>
struct mat3 {
    vec3<T> cols[3];
    
    __host__ __device__ mat3() {}
    __host__ __device__ mat3(const vec3<T>& c0, const vec3<T>& c1, const vec3<T>& c2) {
        cols[0] = c0; cols[1] = c1; cols[2] = c2;
    }
    __host__ __device__ mat3(T m00, T m10, T m20,
                             T m01, T m11, T m21,
                             T m02, T m12, T m22) {
        cols[0] = vec3<T>(m00, m10, m20);
        cols[1] = vec3<T>(m01, m11, m21);
        cols[2] = vec3<T>(m02, m12, m22);
    }
    
    __host__ __device__ vec3<T>& operator[](int i) { return cols[i]; }
    __host__ __device__ const vec3<T>& operator[](int i) const { return cols[i]; }
    
    // Matrix-vector multiplication
    __host__ __device__ vec3<T> operator*(const vec3<T>& v) const {
        return vec3<T>(
            cols[0].x * v.x + cols[1].x * v.y + cols[2].x * v.z,
            cols[0].y * v.x + cols[1].y * v.y + cols[2].y * v.z,
            cols[0].z * v.x + cols[1].z * v.y + cols[2].z * v.z
        );
    }
    
    // Matrix-matrix multiplication
    __host__ __device__ mat3<T> operator*(const mat3<T>& m) const {
        return mat3<T>(
            (*this) * m.cols[0],
            (*this) * m.cols[1],
            (*this) * m.cols[2]
        );
    }
    
    // Matrix addition
    __host__ __device__ mat3<T> operator+(const mat3<T>& m) const {
        return mat3<T>(
            cols[0] + m.cols[0],
            cols[1] + m.cols[1],
            cols[2] + m.cols[2]
        );
    }
    
    __host__ __device__ mat3<T>& operator+=(const mat3<T>& m) {
        cols[0] += m.cols[0];
        cols[1] += m.cols[1];
        cols[2] += m.cols[2];
        return *this;
    }
    
    // Scalar multiplication
    __host__ __device__ mat3<T> operator*(T s) const {
        return mat3<T>(cols[0] * s, cols[1] * s, cols[2] * s);
    }
};

/**
 * 4x4 matrix type (column-major)
 */
template <typename T>
struct mat4 {
    vec4<T> cols[4];
    
    __host__ __device__ mat4() {}
    __host__ __device__ mat4(const vec4<T>& c0, const vec4<T>& c1, 
                             const vec4<T>& c2, const vec4<T>& c3) {
        cols[0] = c0; cols[1] = c1; cols[2] = c2; cols[3] = c3;
    }
    
    __host__ __device__ vec4<T>& operator[](int i) { return cols[i]; }
    __host__ __device__ const vec4<T>& operator[](int i) const { return cols[i]; }
    
    // Matrix-vector multiplication
    __host__ __device__ vec4<T> operator*(const vec4<T>& v) const {
        return vec4<T>(
            cols[0].x * v.x + cols[1].x * v.y + cols[2].x * v.z + cols[3].x * v.w,
            cols[0].y * v.x + cols[1].y * v.y + cols[2].y * v.z + cols[3].y * v.w,
            cols[0].z * v.x + cols[1].z * v.y + cols[2].z * v.z + cols[3].z * v.w,
            cols[0].w * v.x + cols[1].w * v.y + cols[2].w * v.z + cols[3].w * v.w
        );
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Dot product of two vec2
 */
template <typename T>
__host__ __device__ T dot(const vec2<T>& a, const vec2<T>& b) {
    return a.x * b.x + a.y * b.y;
}

/**
 * Dot product of two vec3
 */
template <typename T>
__host__ __device__ T dot(const vec3<T>& a, const vec3<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

/**
 * Dot product of two vec4
 */
template <typename T>
__host__ __device__ T dot(const vec4<T>& a, const vec4<T>& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/**
 * Outer product of two vec3 -> mat3
 */
template <typename T>
__host__ __device__ mat3<T> outer_product(const vec3<T>& a, const vec3<T>& b) {
    return mat3<T>(
        a.x * b.x, a.y * b.x, a.z * b.x,  // column 0
        a.x * b.y, a.y * b.y, a.z * b.y,  // column 1
        a.x * b.z, a.y * b.z, a.z * b.z   // column 2
    );
}

/**
 * Transpose of mat3
 */
template <typename T>
__host__ __device__ mat3<T> transpose(const mat3<T>& m) {
    return mat3<T>(
        m.cols[0].x, m.cols[1].x, m.cols[2].x,
        m.cols[0].y, m.cols[1].y, m.cols[2].y,
        m.cols[0].z, m.cols[1].z, m.cols[2].z
    );
}

/**
 * Identity matrix
 */
template <typename T>
__host__ __device__ mat3<T> identity3() {
    return mat3<T>(
        T(1), T(0), T(0),
        T(0), T(1), T(0),
        T(0), T(0), T(1)
    );
}

} // namespace r2splat


#endif // R2SPLAT_TYPES_CUH
