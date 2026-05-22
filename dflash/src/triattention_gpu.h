// triattention_gpu.h — Lightweight wrapper for GPU compaction functions
//
// This header provides a simple interface to call GPU compaction without
// including full CUDA/HIP headers from .cpp files. No GPU runtime headers
// are included here.

#pragma once

#include <stdint.h>
#include <stddef.h>

namespace dflash27b {
namespace triattention {

// GPU compaction result
enum class GPUCompactResult {
    SUCCESS,
    GPU_NOT_AVAILABLE,
    ALLOC_FAILED,
    COPY_FAILED,
    LAUNCH_FAILED
};

// Check if GPU compaction is available
bool is_gpu_compaction_available();

// Compact a single KV head on GPU
GPUCompactResult gpu_compact_kv_head(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len);

// Compact the tria_k_pre_rope buffer on GPU
GPUCompactResult gpu_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start);

}  // namespace triattention
}  // namespace dflash27b