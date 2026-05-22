// triattention_kernels.cuh — GPU kernel declarations for TriAttention scoring
//
// HIP/CUDA kernels for GPU-resident TriAttention KV compression.
// Eliminates the GPU→CPU→GPU round-trip bottleneck.
//
// This header provides host-side function declarations that are implemented
// in triattention_kernels.cu (CUDA) or triattention_kernels.hip.cu (HIP).

#pragma once

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)

#include <stdint.h>
#include <stddef.h>

#if defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
namespace dflash27b {
namespace triattention {
using gpuStream_t = cudaStream_t;
}  // namespace triattention
}  // namespace dflash27b
#elif defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
namespace dflash27b {
namespace triattention {
using gpuStream_t = hipStream_t;
}  // namespace triattention
}  // namespace dflash27b
#endif

namespace dflash27b {
namespace triattention {

// ── Kernel launch configuration ────────────────────────────────────────

// Threads per block for scoring kernel
#define TRIA_SCORING_BLOCK_SIZE 64

// Maximum frequency count supported
#define TRIA_MAX_FREQ_COUNT 128

// Number of geometric offsets (TRIA_N_OFFSETS = 17)
#define TRIA_N_OFFSETS 17

// ── GPU-resident scoring ────────────────────────────────────────────────
//
// Performs TriAttention scoring entirely on GPU without CPU round-trip.
//
// Algorithm (eq 6-11 and eq 12-13 from TriAttention paper):
//   1. Convert bf16 pre-RoPE K to f32 real/imag on GPU
//   2. For each query head:
//      - Compute rel = q_mean * conj(k) per frequency band
//      - Compute ka = ||k|| per frequency band
//      - Accumulate trig sum over all offsets and frequencies
//   3. Max-pool across GQA groups
//   4. Average across full-attention layers
//
// Kernel design:
//   - Grid: [n_kv_heads * seq_len] blocks, 64 threads each
//   - Per-block: process one (kv_head, position) pair across all layers
//   - Shared memory: q_mean_real[fc], q_mean_imag[fc], omega[fc], offsets
//
// Parameters:
//   k_bf16_gpu - GPU bf16 buffer [tensor_head_dim, max_ctx, n_head_kv]
//   d_q_stats - GPU buffer of Q stats [n_full_attn][fc][3]
//   d_omega - GPU buffer of omega frequencies [fc]
//   d_key_pos - GPU array of positions [seq_len]
//   cur_pos - current position (trigger point)
//   n_full_attn - number of full-attention layers
//   n_kv_heads - number of KV heads
//   tensor_head_dim - tensor head dimension (e.g., 256)
//   fc - frequency count (head_dim / 2, e.g., 64)
//   seq_len - sequence length
//   kv_start - start of KV range
//   d_scores_out - GPU output array [seq_len] for combined scores
//   gqa - groups per attention head (num_heads / num_kv_heads)
//   stream - GPU stream
//
// Returns true on success, false on error.

#if defined(GGML_USE_CUDA)
bool ggml_cuda_tria_score(
    const void * k_bf16_gpu,
    const float * d_q_stats,
    const float * d_omega,
    const int   * d_key_pos,
    int           cur_pos,
    int           n_full_attn,
    int           n_kv_heads,
    int           tensor_head_dim,
    int           fc,
    int           seq_len,
    int           kv_start,
    float       * d_scores_out,
    int           gqa,
    cudaStream_t  stream);
#elif defined(GGML_USE_HIP)
bool ggml_hip_tria_score(
    const void * k_bf16_gpu,
    const float * d_q_stats,
    const float * d_omega,
    const int   * d_key_pos,
    int           cur_pos,
    int           n_full_attn,
    int           n_kv_heads,
    int           tensor_head_dim,
    int           fc,
    int           seq_len,
    int           kv_start,
    float       * d_scores_out,
    int           gqa,
    hipStream_t   stream);
#endif

// ── Top-K selection on GPU ─────────────────────────────────────────────
//
// Selects top-K positions by score using GPU-based sorting.
//
// Parameters:
//   d_scores - GPU array [seq_len] of combined scores
//   seq_len - sequence length
//   k - number of positions to select
//   d_topk_out - GPU output array [k] for selected positions
//   d_count_out - GPU output [1] for actual count
//   stream - GPU stream
//
// Returns true on success, false on error.

#if defined(GGML_USE_CUDA)
bool ggml_cuda_tria_topk(
    const float * d_scores,
    int           seq_len,
    int           k,
    int         * d_topk_out,
    int         * d_count_out,
    cudaStream_t  stream);
#elif defined(GGML_USE_HIP)
bool ggml_hip_tria_topk(
    const float * d_scores,
    int           seq_len,
    int           k,
    int         * d_topk_out,
    int         * d_count_out,
    hipStream_t   stream);
#endif

// ── GPU KV compaction kernel ───────────────────────────────────────────
//
// Compacts KV cache in-place on GPU using the selected keep_indices.
//
// Parameters:
//   cache_data - GPU buffer [n_kv_heads, max_ctx, head_bytes]
//   keep_indices - GPU array of positions to keep [actual_keep]
//   actual_keep - number of positions to keep
//   kv_head - which KV head
//   max_ctx - maximum context length
//   kv_start - start of KV range
//   head_bytes - bytes per position per head
//   stream - GPU stream
//
// Returns true on success, false on error.

#if defined(GGML_USE_CUDA)
bool ggml_cuda_tria_compact_kv(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len,
    cudaStream_t  stream);
#elif defined(GGML_USE_HIP)
bool ggml_hip_tria_compact_kv(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len,
    hipStream_t   stream);
#endif

// ── GPU KV compaction for tria_k_pre_rope buffer (bf16) ───────────────────
//
// Compacts the pre-RoPE K buffer in-place on GPU.
//
// Parameters:
//   tria_data - GPU bf16 buffer [n_kv_heads, max_ctx, tensor_head_dim * sizeof(uint16_t)]
//   keep_indices - GPU array of positions to keep
//   actual_keep - number of positions to keep
//   tensor_head_dim - elements per position per head
//   max_ctx - maximum context length
//   n_kv_heads - number of KV heads
//   kv_start - start of KV range
//   stream - GPU stream
//
// Returns true on success, false on error.

#if defined(GGML_USE_CUDA)
bool ggml_cuda_tria_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start,
    cudaStream_t  stream);
#elif defined(GGML_USE_HIP)
bool ggml_hip_tria_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start,
    hipStream_t   stream);
#endif

}  // namespace triattention
}  // namespace dflash27b

#endif // GGML_USE_CUDA || GGML_USE_HIP