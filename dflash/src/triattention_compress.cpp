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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(GGML_USE_CUDA) && !defined(GGML_USE_HIP)
#include <cuda_runtime.h>
#define HAS_GPU 1
#define gpuGetDevice   cudaGetDevice
#define gpuSetDevice   cudaSetDevice
#define gpuMemcpy      cudaMemcpy
#define gpuMemcpyDtoH   cudaMemcpyDeviceToHost
#define gpuMemcpyHtoD   cudaMemcpyHostToDevice
#elif defined(GGML_USE_HIP)
#include <hip/hip_runtime.h>
#define HAS_GPU 1
#define gpuGetDevice   hipGetDevice
#define gpuSetDevice   hipSetDevice
#define gpuMemcpy       hipMemcpy
#define gpuMemcpyDtoH   hipMemcpyDeviceToHost
#define gpuMemcpyHtoD   hipMemcpyHostToDevice
#else
#define HAS_GPU 0
#endif

namespace dflash27b {

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
// Copies positions in keep_indices order: position keep_indices[i] goes to position i
// Uses overlapping-safe copy (handles in-place compaction correctly).
static void compact_kv_head_positions(
    void * cache_data,
    int kv_head,
    int max_ctx,
    int kv_start,
    const std::vector<int> & keep_indices,
    int actual_keep,
    size_t head_bytes)
{
    if (!cache_data || actual_keep == 0) return;

    const size_t head_stride = max_ctx * head_bytes;

    for (int ki = 0; ki < actual_keep; ki++) {
        const int src_pos = keep_indices[ki];  // relative to kv_start
        const int dst_pos = ki;

        if (src_pos == dst_pos) continue;

        const size_t src_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + src_pos) * head_bytes;
        const size_t dst_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + dst_pos) * head_bytes;

        std::vector<uint8_t> tmp(head_bytes);
        std::memcpy(tmp.data(), (char*)cache_data + src_offset, head_bytes);
        std::memcpy((char*)cache_data + dst_offset, tmp.data(), head_bytes);
    }
}

// ─── Helper: Compact tria_k_pre_rope buffer ─────────────────────────────
// Compacts the pre-RoPE K buffer in-place to match the KV cache compaction.
// This ensures that subsequent forward passes read the correct pre-RoPE K data.
static void compact_tria_k_pre_rope(
    void * tria_data,
    int head_dim,
    int max_ctx,
    int n_head_kv,
    const std::vector<int> & keep_indices,
    int kv_start,
    int actual_keep)
{
    if (!tria_data || actual_keep == 0) return;

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
    int max_ctx,
    int kv_start,
    int cur_pos,
    int gpu_id,
    float keep_ratio,
    int * n_kept_out,
    enum ggml_type k_type,
    enum ggml_type v_type)
{
    // Get element sizes from ggml types
    const size_t k_bytes = head_dim * ggml_type_size(k_type) / ggml_blck_size(k_type);
    const size_t v_bytes = head_dim * ggml_type_size(v_type) / ggml_blck_size(v_type);

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

#if HAS_GPU
    // Get current GPU device
    int current_device = 0;
    gpuGetDevice(&current_device);

    // Set to requested GPU if different
    if (gpu_id >= 0 && gpu_id != current_device) {
        gpuSetDevice(gpu_id);
    }

    // Read pre-RoPE K from GPU if buffer is provided
    std::vector<float> k_real;
    std::vector<float> k_imag;
    if (k_pre_rope_gpu) {
        // Allocate CPU buffer for pre-RoPE K: [head_dim, seq_len, n_head_kv]
        const size_t nelems = (size_t)head_dim * seq_len * n_head_kv;
        std::vector<uint16_t> k_bf16(nelems);

        // Copy from GPU (only the relevant slice)
        // k_pre_rope_gpu is [head_dim, max_ctx, n_head_kv]
        const size_t src_offset = (size_t)kv_start * head_dim * n_head_kv * sizeof(uint16_t);
        const size_t copy_bytes = (size_t)head_dim * seq_len * n_head_kv * sizeof(uint16_t);
        gpuMemcpy(k_bf16.data(), (char*)k_pre_rope_gpu + src_offset, copy_bytes,
                  gpuMemcpyDtoH);

        // Convert bf16 to f32 and split into real/imag halves
        // Layout: [head_dim, seq_len, n_head_kv] -> split into [fc, seq_len, n_head_kv] for real and imag
        k_real.resize((size_t)fc * seq_len * n_head_kv);
        k_imag.resize((size_t)fc * seq_len * n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            for (int s = 0; s < seq_len; s++) {
                for (int f = 0; f < fc; f++) {
                    const size_t src_idx = ((size_t)h * seq_len + s) * head_dim + f;
                    const size_t real_idx = ((size_t)h * seq_len + s) * fc + f;
                    const size_t imag_idx = ((size_t)h * seq_len + s) * fc + fc + f;

                    // Real half: elements [0, fc)
                    k_real[real_idx] = bf16_to_f32(k_bf16[src_idx]);

                    // Imag half: elements [fc, 2*fc)
                    k_imag[imag_idx] = bf16_to_f32(k_bf16[src_idx + fc]);
                }
            }
        }
    }

    // Restore original device
    if (gpu_id >= 0 && gpu_id != current_device) {
        gpuSetDevice(current_device);
    }

    // Build key_pos array: positions [kv_start, kv_start+1, ..., cur_pos-1]
    std::vector<int> key_pos(seq_len);
    for (int i = 0; i < seq_len; i++) {
        key_pos[i] = kv_start + i;
    }

    // Score each full-attention layer via tria_score_kv_head()
    // For each layer, we get per-position scores aggregated across KV heads
    std::vector<float> combined_scores(seq_len, 0.0f);

    for (int fa_layer = 0; fa_layer < n_full_attn; fa_layer++) {
        // Map FA layer index to global layer index (every 4th layer)
        const int global_layer = fa_layer * 4 + 3;  // layers 3,7,11,...
        if (global_layer >= (int)stats->num_layers) continue;

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
#if HAS_GPU
    // Set device for compression operations
    if (gpu_id >= 0) {
        gpuSetDevice(gpu_id);
    }

    std::fprintf(stderr, "[TriAttention] Compacting %d FA layers, %d KV heads, k_bytes=%zu, v_bytes=%zu\n",
                n_full_attn, n_head_kv, k_bytes, v_bytes);

    for (int fa_layer = 0; fa_layer < n_full_attn; fa_layer++) {
        if (!attn_k[fa_layer] || !attn_v[fa_layer]) continue;

        for (int h = 0; h < n_head_kv; h++) {
            // K cache compaction
            compact_kv_head_positions(attn_k[fa_layer], h, max_ctx, kv_start,
                                     keep_indices, actual_keep, k_bytes);

            // V cache compaction
            compact_kv_head_positions(attn_v[fa_layer], h, max_ctx, kv_start,
                                     keep_indices, actual_keep, v_bytes);
        }
    }

    // Also compact the tria_k_pre_rope buffer to maintain consistency
    if (k_pre_rope_gpu) {
        std::fprintf(stderr, "[TriAttention] Compacting tria_k_pre_rope buffer\n");
        compact_tria_k_pre_rope(k_pre_rope_gpu, head_dim, max_ctx, n_head_kv,
                               keep_indices, kv_start, actual_keep);
    }
#endif

    const int new_cur_pos = kv_start + actual_keep;
    if (n_kept_out) *n_kept_out = new_cur_pos;

    std::fprintf(stderr, "[TriAttention] Compressed %d -> %d positions (kept %.1f%%)\n",
                seq_len, actual_keep, 100.0f * actual_keep / seq_len);

    return true;
}

}  // namespace dflash27b
