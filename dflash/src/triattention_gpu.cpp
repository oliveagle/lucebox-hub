// triattention_gpu.cpp — Implementation of GPU compaction wrapper

#include "triattention_gpu.h"

#if defined(GGML_USE_HIP)
#include "triattention_kernels.cuh"
#include <hip/hip_runtime_api.h>
#elif defined(GGML_USE_CUDA)
#include "triattention_kernels.cuh"
#include <cuda_runtime.h>
#else
// No GPU compaction available
#endif

#include <cstdio>

namespace dflash27b {
namespace triattention {

bool is_gpu_compaction_available() {
#if (defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)) && !defined(GGML_USE_HIP_NO_GPU)
    return true;
#else
    return false;
#endif
}

GPUCompactResult gpu_compact_kv_head(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len)
{
#if defined(GGML_USE_HIP)
    // HIP: Allocate device keep_indices
    int * d_keep_indices = nullptr;
    hipError_t err = hipMalloc((void**)&d_keep_indices, actual_keep * sizeof(int));
    if (err != hipSuccess) {
        return GPUCompactResult::ALLOC_FAILED;
    }

    err = hipMemcpy(d_keep_indices, keep_indices,
                     actual_keep * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        (void)hipFree(d_keep_indices);
        return GPUCompactResult::COPY_FAILED;
    }

    bool success = ggml_hip_tria_compact_kv(
        cache_data, d_keep_indices, actual_keep,
        kv_head, max_ctx, kv_start, head_bytes, seq_len, nullptr);

    (void)hipFree(d_keep_indices);
    return success ? GPUCompactResult::SUCCESS : GPUCompactResult::LAUNCH_FAILED;
#elif defined(GGML_USE_CUDA)
    // CUDA: Allocate device keep_indices
    int * d_keep_indices = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_keep_indices, actual_keep * sizeof(int));
    if (err != cudaSuccess) {
        return GPUCompactResult::ALLOC_FAILED;
    }

    err = cudaMemcpy(d_keep_indices, keep_indices,
                     actual_keep * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_keep_indices);
        return GPUCompactResult::COPY_FAILED;
    }

    bool success = ggml_cuda_tria_compact_kv(
        cache_data, d_keep_indices, actual_keep,
        kv_head, max_ctx, kv_start, head_bytes, seq_len, nullptr);

    cudaFree(d_keep_indices);
    return success ? GPUCompactResult::SUCCESS : GPUCompactResult::LAUNCH_FAILED;
#else
    (void)cache_data; (void)keep_indices; (void)actual_keep;
    (void)kv_head; (void)max_ctx; (void)kv_start;
    (void)head_bytes; (void)seq_len;
    return GPUCompactResult::GPU_NOT_AVAILABLE;
#endif
}

GPUCompactResult gpu_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start)
{
#if defined(GGML_USE_HIP)
    // HIP: Allocate device keep_indices
    int * d_keep_indices = nullptr;
    hipError_t err = hipMalloc((void**)&d_keep_indices, actual_keep * sizeof(int));
    if (err != hipSuccess) {
        return GPUCompactResult::ALLOC_FAILED;
    }

    err = hipMemcpy(d_keep_indices, keep_indices,
                     actual_keep * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        (void)hipFree(d_keep_indices);
        return GPUCompactResult::COPY_FAILED;
    }

    bool success = ggml_hip_tria_compact_tria_bf16(
        tria_data, d_keep_indices, actual_keep,
        tensor_head_dim, max_ctx, n_kv_heads, kv_start, nullptr);

    (void)hipFree(d_keep_indices);
    return success ? GPUCompactResult::SUCCESS : GPUCompactResult::LAUNCH_FAILED;
#elif defined(GGML_USE_CUDA)
    // CUDA: Allocate device keep_indices
    int * d_keep_indices = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_keep_indices, actual_keep * sizeof(int));
    if (err != cudaSuccess) {
        return GPUCompactResult::ALLOC_FAILED;
    }

    err = cudaMemcpy(d_keep_indices, keep_indices,
                     actual_keep * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_keep_indices);
        return GPUCompactResult::COPY_FAILED;
    }

    bool success = ggml_cuda_tria_compact_tria_bf16(
        tria_data, d_keep_indices, actual_keep,
        tensor_head_dim, max_ctx, n_kv_heads, kv_start, nullptr);

    cudaFree(d_keep_indices);
    return success ? GPUCompactResult::SUCCESS : GPUCompactResult::LAUNCH_FAILED;
#else
    (void)tria_data; (void)keep_indices; (void)actual_keep;
    (void)tensor_head_dim; (void)max_ctx; (void)n_kv_heads;
    (void)kv_start;
    return GPUCompactResult::GPU_NOT_AVAILABLE;
#endif
}

}  // namespace triattention
}  // namespace dflash27b
