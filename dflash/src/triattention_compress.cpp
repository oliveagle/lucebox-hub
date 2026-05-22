// triattention_compress.cpp — TriAttention KV compression implementation for DFlash
//
// Implements CPU-side KV cache compression using TriAttention frequency scoring.
// This runs outside the ggml compute graph, triggered at decode intervals.
//
// Flow:
//   1. Read pre-RoPE K from GPU buffer (tria_k_pre_rope)
//   2. Run frequency-domain scoring per KV head on CPU via tria_score_kv_head()
//   3. Aggregate scores across GQA groups (max pooling per layer)
//   4. Average scores across full-attention layers
//   5. Select top-K positions by score (with recent window preservation)
//   6. Compact K and V caches in-place (move kept positions to front)
//   7. Update cur_pos to new compressed length

#include "triattention_runner.h"

#if defined(DFLASH27B_TRIATTENTION_ENABLED)
#include "triattention.h"
#endif

#include "triattention_gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define HAS_GPU 1
#define GPU_SUCCESS hipSuccess
#define gpuGetDevice   hipGetDevice
#define gpuSetDevice   hipSetDevice
#define gpuMemcpy      hipMemcpy
#define gpuMemcpyDtoH  hipMemcpyDeviceToHost
#define gpuMemcpyHtoD  hipMemcpyHostToDevice
#define gpuMalloc      hipMalloc
#define gpuFree        hipFree
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetErrorString hipGetErrorString
#elif defined(GGML_USE_CUDA)
#include <cuda_runtime.h>
#define HAS_GPU 1
#define GPU_SUCCESS cudaSuccess
#define gpuGetDevice   cudaGetDevice
#define gpuSetDevice   cudaSetDevice
#define gpuMemcpy      cudaMemcpy
#define gpuMemcpyDtoH  cudaMemcpyDeviceToHost
#define gpuMemcpyHtoD  cudaMemcpyHostToDevice
#define gpuMalloc      cudaMalloc
#define gpuFree        cudaFree
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetErrorString cudaGetErrorString
#else
#define HAS_GPU 0
#define gpuGetDevice(...)  ((void)0)
#define gpuSetDevice(...)  ((void)0)
#define gpuMemcpy(...)     ((void)0)
#define gpuMemcpyDtoH(...) ((void)0)
#define gpuMemcpyHtoD(...) ((void)0)
#define gpuMalloc(...)     ((void*)(0))
#define gpuFree(...)       ((void)0)
#endif

namespace dflash27b {

#if defined(HAS_GPU) && HAS_GPU
// Helper macro to check GPU API call results
#define GPU_CHECK(call, msg) do { \
    auto err = (call); \
    if (err != GPU_SUCCESS) { \
        std::fprintf(stderr, "[TriAttention] GPU error: %s: %s\n", (msg), gpuGetErrorString(err)); \
        return false; \
    } \
} while(0)
#else
#define GPU_CHECK(call, msg) ((void)0)
#endif

// ── bf16 to f32 conversion ──────────────────────────────────────────────

// Convert bfloat16 (stored as uint16_t) to float32 via union bit-cast
static inline float bf16_to_f32(uint16_t h) {
    union u32 { uint32_t bits; float f; } conv;
    conv.bits = static_cast<uint32_t>(h) << 16;
    return conv.f;
}

// ── Top-K selection by score (descending) with window preservation ─────

static std::vector<int> select_top_k_with_window(
    const std::vector<float> & scores,
    int k,
    int window_size,
    int seq_start)
{
    const int n = (int)scores.size();
    const int total_positions = seq_start + n;
    k = std::min(k, n);

    // Build index array
    std::vector<std::pair<float, int>> indexed(n);
    for (int i = 0; i < n; i++) {
        indexed[i] = {scores[i], i};
    }

    // Partial sort descending by score
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    // Collect top-K indices (relative to seq_start)
    std::vector<int> result;
    result.reserve(k);
    for (int i = 0; i < k; i++) {
        const int global_pos = seq_start + indexed[i].second;
        // Preserve recent window: always keep positions within window_size of current
        if (global_pos >= total_positions - window_size) {
            // Add to front (recent positions take priority)
            result.insert(result.begin(), indexed[i].second);
        } else {
            result.push_back(indexed[i].second);
        }
    }

    // Deduplicate and sort for sequential access
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());

    // Ensure window positions are always included
    const int window_start = std::max(seq_start, total_positions - window_size);
    for (int pos = window_start; pos < total_positions; pos++) {
        const int rel_pos = pos - seq_start;
        if (std::find(result.begin(), result.end(), rel_pos) == result.end()) {
            result.push_back(rel_pos);
        }
    }
    std::sort(result.begin(), result.end());

    return result;
}

// ─── Helper: Copy positions from src to dst for a single KV head ─────
// Uses GPU kernel for compaction when available (eliminates DtoH/HtoD round trips).
// Returns false on GPU error, true on success.
static bool compact_kv_head_positions(
    void * cache_data,
    int kv_head,
    int max_ctx,
    int kv_start,
    const std::vector<int> & keep_indices,
    int actual_keep,
    size_t head_bytes)
{
    if (!cache_data || actual_keep == 0) return true;

    // Try GPU compaction first
    if (dflash27b::triattention::is_gpu_compaction_available()) {
        auto result = dflash27b::triattention::gpu_compact_kv_head(
            cache_data, keep_indices.data(), actual_keep,
            kv_head, max_ctx, kv_start, head_bytes,
            (int)keep_indices.size());
        if (result == dflash27b::triattention::GPUCompactResult::SUCCESS) {
            return true;
        }
        // Fall through to CPU path on failure
    }

    // CPU path: direct memcpy
    const size_t head_stride = max_ctx * head_bytes;
    for (int ki = 0; ki < actual_keep; ki++) {
        const int src_pos = keep_indices[ki];
        const int dst_pos = ki;
        if (src_pos == dst_pos) continue;

        const size_t src_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + src_pos) * head_bytes;
        const size_t dst_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + dst_pos) * head_bytes;
        std::vector<uint8_t> tmp(head_bytes);
        std::memcpy(tmp.data(), (char*)cache_data + src_offset, head_bytes);
        std::memcpy((char*)cache_data + dst_offset, tmp.data(), head_bytes);
    }
    return true;
}

// ─── Helper: Compact tria_k_pre_rope buffer ─────────────────────────────
// Uses GPU kernel for compaction when available.
// Returns false on GPU error, true on success.
static bool compact_tria_k_pre_rope(
    void * tria_data,
    int head_dim,
    int max_ctx,
    int n_head_kv,
    const std::vector<int> & keep_indices,
    int kv_start,
    int actual_keep)
{
    if (!tria_data || actual_keep == 0) return true;

    // Try GPU compaction first
    if (dflash27b::triattention::is_gpu_compaction_available()) {
        auto result = dflash27b::triattention::gpu_compact_tria_bf16(
            tria_data, keep_indices.data(), actual_keep,
            head_dim, max_ctx, n_head_kv, kv_start);
        if (result == dflash27b::triattention::GPUCompactResult::SUCCESS) {
            return true;
        }
        // Fall through to CPU path on failure
    }

    // CPU path: direct memcpy
    const size_t head_bytes = (size_t)head_dim * sizeof(uint16_t);
    const size_t head_stride = max_ctx * head_bytes;
    for (int h = 0; h < n_head_kv; h++) {
        for (int ki = 0; ki < actual_keep; ki++) {
            const int src_pos = keep_indices[ki];
            const int dst_pos = ki;
            if (src_pos == dst_pos) continue;

            const size_t src_offset = ((size_t)h * head_stride) + ((size_t)kv_start + src_pos) * head_bytes;
            const size_t dst_offset = ((size_t)h * head_stride) + ((size_t)kv_start + dst_pos) * head_bytes;
            std::vector<uint8_t> tmp(head_bytes);
            std::memcpy(tmp.data(), (char*)tria_data + src_offset, head_bytes);
            std::memcpy((char*)tria_data + dst_offset, tmp.data(), head_bytes);
        }
    }
    return true;
}

// ── Main KV compression function ────────────────────────────────────────

bool tria_kv_compress(
    const struct tria_stats * stats,
    void * k_pre_rope_gpu,
    void * attn_k[],
    void * attn_v[],
    int n_full_attn,
    int n_head_kv,
    int head_dim,
    int tensor_head_dim,
    int max_ctx,
    int kv_start,
    int cur_pos,
    int gpu_id,
    float keep_ratio,
    int * n_kept_out,
    enum ggml_type k_type,
    enum ggml_type v_type)
{
    // Get element sizes from ggml types using tensor_head_dim (the actual K/V tensor layout)
    const size_t k_bytes = tensor_head_dim * ggml_type_size(k_type) / ggml_blck_size(k_type);
    const size_t v_bytes = tensor_head_dim * ggml_type_size(v_type) / ggml_blck_size(v_type);

    std::fprintf(stderr, "[TriAttention] tria_kv_compress: n_full_attn=%d n_head_kv=%d head_dim=%d max_ctx=%d kv_start=%d cur_pos=%d seq_len=%d\n",
                n_full_attn, n_head_kv, head_dim, max_ctx, kv_start, cur_pos, cur_pos - kv_start);
    std::fprintf(stderr, "[TriAttention] stats: num_layers=%u num_heads=%u num_kv_heads=%u head_dim=%u freq_count=%u\n",
                stats->num_layers, stats->num_heads, stats->num_kv_heads, stats->head_dim, stats->freq_count);
    std::fprintf(stderr, "[TriAttention] k_bytes=%zu, v_bytes=%zu, keep_ratio=%.2f\n", k_bytes, v_bytes, keep_ratio);
    std::fflush(stderr);

    if (!stats || cur_pos <= 0) {
        if (n_kept_out) *n_kept_out = cur_pos;
        return true;
    }

    const int seq_len = cur_pos - kv_start;
    if (seq_len <= 0) {
        if (n_kept_out) *n_kept_out = cur_pos;
        return true;
    }

    // Calculate actual keep count from ratio
    const int n_keep = std::max(1, (int)(seq_len * keep_ratio));
    const int fc = (int)stats->freq_count;

    std::fprintf(stderr, "[TriAttention] n_keep=%d, fc=%d\n", n_keep, fc);

#if defined(HAS_GPU) && HAS_GPU
    // ─── GPU + CPU PATH: Optimized implementation ───────────────────────
    // 1. GPU: Copy bf16 from GPU to CPU (large transfer)
    // 2. CPU: bf16→f32 conversion (triple loop, slow but manageable)
    // 3. CPU: tria_score_kv_head scoring (main bottleneck)
    // 4. CPU: top-K selection with window preservation
    // 5. CPU + GPU: KV cache compaction (batched reads/writes)
    //
    // TODO: Add pure GPU path for bf16→f32 and scoring to avoid DtoH bottleneck

    // Get current GPU device
    int current_device = 0;
    GPU_CHECK(gpuGetDevice(&current_device), "gpuGetDevice");

    // Set to requested GPU if different
    if (gpu_id >= 0 && gpu_id != current_device) {
        GPU_CHECK(gpuSetDevice(gpu_id), "gpuSetDevice");
    }

    // Read pre-RoPE K from GPU if buffer is provided
    std::vector<float> k_real;
    std::vector<float> k_imag;
    if (k_pre_rope_gpu) {
        // The tria_k_pre_rope buffer has head_dim=tensor_head_dim (e.g., 256)
        // But only the first 'head_dim' (e.g., 64) elements contain RoPE data
        // We need to copy the full tensor_head_dim, then extract the RoPE portion

        // Allocate CPU buffer for full pre-RoPE K: [tensor_head_dim, seq_len, n_head_kv]
        const size_t full_nelems = (size_t)tensor_head_dim * seq_len * n_head_kv;
        std::vector<uint16_t> k_bf16(full_nelems);

        // Copy from GPU (only the relevant slice)
        // k_pre_rope_gpu is [tensor_head_dim, max_ctx, n_head_kv]
        const size_t src_offset = (size_t)kv_start * tensor_head_dim * n_head_kv * sizeof(uint16_t);
        const size_t copy_bytes = (size_t)tensor_head_dim * seq_len * n_head_kv * sizeof(uint16_t);

        std::fprintf(stderr, "[TriAttention] Reading tria_k_pre_rope: offset=%zu, bytes=%zu\n", src_offset, copy_bytes);

        GPU_CHECK(gpuMemcpy(k_bf16.data(), (char*)k_pre_rope_gpu + src_offset, copy_bytes, gpuMemcpyDtoH),
                  "gpuMemcpy DtoH (reading tria_k_pre_rope)");

        // Synchronize after GPU copy to ensure completion
        GPU_CHECK(gpuDeviceSynchronize(), "gpuDeviceSynchronize after reading tria_k_pre_rope");

        std::fprintf(stderr, "[TriAttention] GPU copy done, starting conversion\n");

        // Convert bf16 to f32 and split into real/imag halves
        // Layout: [tensor_head_dim, seq_len, n_head_kv] -> split into [fc, seq_len, n_head_kv] for real and imag
        // Only the first 'head_dim' elements of the tensor_head_dim contain RoPE data
        k_real.resize((size_t)fc * seq_len * n_head_kv);
        k_imag.resize((size_t)fc * seq_len * n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int f = 0; f < fc; f++) {
                    // Read from full tensor (tensor_head_dim stride)
                    const size_t src_idx = ((size_t)h * seq_len + s) * tensor_head_dim + f;
                    // Real and imag are separate arrays, so same [h,s,f] indexing
                    const size_t real_idx = ((size_t)h * seq_len + s) * fc + f;
                    const size_t imag_idx = ((size_t)h * seq_len + s) * fc + f;

                    // Real half: elements [0, fc) from the tensor
                    k_real[real_idx] = bf16_to_f32(k_bf16[src_idx]);

                    // Imag half: elements [fc, 2*fc) from the tensor
                    k_imag[imag_idx] = bf16_to_f32(k_bf16[src_idx + fc]);
                }
            }
        }

        std::fprintf(stderr, "[TriAttention] Finished converting tria_k_pre_rope to real/imag\n");
    }

    // Restore original device
    if (gpu_id >= 0 && gpu_id != current_device) {
        GPU_CHECK(gpuSetDevice(current_device), "gpuSetDevice (restore)");
    }

    // Build key_pos array: positions [kv_start, kv_start+1, ..., cur_pos-1]
    std::vector<int> key_pos(seq_len);
    for (int i = 0; i < seq_len; i++) {
        key_pos[i] = kv_start + i;
    }

    // Score each full-attention layer via tria_score_kv_head()
    // For each layer, we get per-position scores aggregated across KV heads
    std::vector<float> combined_scores(seq_len, 0.0f);

    std::fprintf(stderr, "[TriAttention] Starting scoring loop: n_full_attn=%d\n", n_full_attn);

    for (int fa_layer = 0; fa_layer < n_full_attn; fa_layer++) {
        // Map FA layer index to global layer index (every 4th layer)
        const int global_layer = fa_layer * 4 + 3;  // layers 3,7,11,...
        if (global_layer >= (int)stats->num_layers) continue;

        // std::fprintf(stderr, "[TriAttention] Scoring FA layer %d (global %d)\n", fa_layer, global_layer);

        // Score each KV head and aggregate (max pooling)
        std::vector<float> layer_scores(seq_len, 0.0f);
        for (int kv_h = 0; kv_h < n_head_kv; kv_h++) {
            std::vector<float> kv_scores(seq_len, 0.0f);

            // Call the real TriAttention scoring function
            tria_score_kv_head(stats,
                               k_real.data(), k_imag.data(),
                               key_pos.data(),
                               cur_pos,
                               seq_len,
                               global_layer,
                               kv_h,
                               kv_scores.data());

            // Max-pool across KV heads for this layer
            for (int s = 0; s < seq_len; s++) {
                layer_scores[s] = std::max(layer_scores[s], kv_scores[s]);
            }
        }

        // Accumulate layer scores
        for (int s = 0; s < seq_len; s++) {
            combined_scores[s] += layer_scores[s];
        }
    }

    std::fprintf(stderr, "[TriAttention] Finished scoring all FA layers\n");

    // Average scores across layers
    if (n_full_attn > 0) {
        for (int s = 0; s < seq_len; s++) {
            combined_scores[s] /= n_full_attn;
        }
    }

    // Select top-K positions with window preservation
    const int window_size = g_tria_state.window_size;
    std::vector<int> keep_indices = select_top_k_with_window(
        combined_scores, n_keep, window_size, kv_start);
    const int actual_keep = (int)keep_indices.size();

#else
    // Non-GPU builds just report and skip
    std::fprintf(stderr, "[TriAttention] GPU not available, skipping compression\n");
    if (n_kept_out) *n_kept_out = cur_pos;
    return true;
#endif

    // Compact KV cache for each full-attention layer
#if defined(HAS_GPU) && HAS_GPU
    // Set device for compression operations
    if (gpu_id >= 0) {
        GPU_CHECK(gpuSetDevice(gpu_id), "gpuSetDevice (compaction)");
    }

    std::fprintf(stderr, "[TriAttention] Compacting %d FA layers, %d KV heads, k_bytes=%zu, v_bytes=%zu\n",
                n_full_attn, n_head_kv, k_bytes, v_bytes);

    // Skip compaction if we're keeping everything (no actual compression)
    if (actual_keep >= seq_len) {
        std::fprintf(stderr, "[TriAttention] Skipping compaction (actual_keep=%d >= seq_len=%d)\n", actual_keep, seq_len);
    } else {
        for (int fa_layer = 0; fa_layer < n_full_attn; fa_layer++) {
            if (!attn_k[fa_layer] || !attn_v[fa_layer]) continue;

            std::fprintf(stderr, "[TriAttention] Compacting FA layer %d\n", fa_layer);

            for (int h = 0; h < n_head_kv; h++) {
                // K cache compaction
                if (!compact_kv_head_positions(attn_k[fa_layer], h, max_ctx, kv_start,
                                              keep_indices, actual_keep, k_bytes)) {
                    return false;
                }

                // V cache compaction
                if (!compact_kv_head_positions(attn_v[fa_layer], h, max_ctx, kv_start,
                                              keep_indices, actual_keep, v_bytes)) {
                    return false;
                }
            }
        }

        std::fprintf(stderr, "[TriAttention] Finished compacting FA layers\n");
    }

    // Also compact the tria_k_pre_rope buffer to maintain consistency
    if (k_pre_rope_gpu && actual_keep < seq_len) {
        std::fprintf(stderr, "[TriAttention] Compacting tria_k_pre_rope buffer (tensor_head_dim=%d)\n", tensor_head_dim);
        if (!compact_tria_k_pre_rope(k_pre_rope_gpu, tensor_head_dim, max_ctx, n_head_kv,
                                    keep_indices, kv_start, actual_keep)) {
            return false;
        }
    }
#endif

    const int new_cur_pos = kv_start + actual_keep;
    if (n_kept_out) *n_kept_out = new_cur_pos;

    std::fprintf(stderr, "[TriAttention] Compressed %d -> %d positions (kept %.1f%%)\n",
                seq_len, actual_keep, 100.0f * actual_keep / seq_len);
    std::fprintf(stderr, "[TriAttention] tria_kv_compress: about to return true\n");
    std::fflush(stderr);

    return true;
}

}  // namespace dflash27b
