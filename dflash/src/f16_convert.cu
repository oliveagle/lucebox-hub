// Tiny half-precision → f32 conversion kernels used by the DDtree rollback
// path and the drafter's target_feat widen. We store some tensors
// (ssm_intermediate, target_feat) at 16-bit to halve their memory footprint,
// and widen on read into f32 consumers.
//
// Exposes plain C entry points so test_dflash.cpp can call them without
// pulling in a CUDA compile unit of its own.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

static __global__ void f16_to_f32_kernel(const __half * __restrict__ src,
                                         float * __restrict__ dst,
                                         size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __half2float(src[i]);
    }
}

static __global__ void bf16_to_f32_kernel(const __nv_bfloat16 * __restrict__ src,
                                          float * __restrict__ dst,
                                          size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_elems) {
        dst[i] = __bfloat162float(src[i]);
    }
}

// Q8_0 block: 1 f16 scale + 32 int8 values. Dequantize: val = scale * qi.
struct block_q8_0_dflash {
    __half d;
    int8_t qs[32];
};

static __global__ void q8_0_to_f32_kernel(const void * __restrict__ src,
                                          float * __restrict__ dst,
                                          size_t n_elems) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    const size_t block_idx = i / 32;
    const size_t within    = i % 32;
    const block_q8_0_dflash * b = (const block_q8_0_dflash *)src + block_idx;
    dst[i] = __half2float(b->d) * (float)b->qs[within];
}

extern "C" void dflash27b_launch_f16_to_f32(const void * src,
                                            void * dst,
                                            size_t n_elems,
                                            cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __half *)src, (float *)dst, n_elems);
}

extern "C" void dflash27b_launch_bf16_to_f32(const void * src,
                                             void * dst,
                                             size_t n_elems,
                                             cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        (const __nv_bfloat16 *)src, (float *)dst, n_elems);
}

extern "C" void dflash27b_launch_q8_0_to_f32(const void * src,
                                              void * dst,
                                              size_t n_elems,
                                              cudaStream_t stream) {
    const int threads = 256;
    const int blocks  = (int)((n_elems + threads - 1) / threads);
    q8_0_to_f32_kernel<<<blocks, threads, 0, stream>>>(
        src, (float *)dst, n_elems);
}
