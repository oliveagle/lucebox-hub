// triattention_runner.h — TriAttention runtime integration for DFlash
//
// This file provides a C++ wrapper around the TriAttention C library
// (triattention.c/h) for KV cache compression in DFlash.
//
// INTEGRATION STATUS (2026-05-18):
//   - TriAttention C library is built (libtriattention.a)
//   - Build flags configured (DFLASH27B_TRIATTENTION)
//   - Runtime integration is IN PROGRESS
//
// USAGE (when fully implemented):
//   1. Load stats at init: tria_load(path)
//   2. Every N tokens, score and prune KV cache
//   3. Free stats at shutdown
//
// INTEGRATION POINTS:
//   - qwen35_target_graph.cpp: build_full_attn_block() after KV write
//   - internal.h: add TriAttentionState to TargetCache struct
//
// NOTE: Full integration requires:
//   1. Capturing pre-RoPE K before ggml_rope_multi()
//   2. Calling tria_score_kv_head() for each KV head
//   3. Implementing KV compaction based on scores

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"

// Include the full TriAttention C library header for tria_stats definition
#include "triattention.h"

namespace dflash27b {

// TriAttention runtime state.
// Manages the lifecycle of the tria_stats pointer and provides
// C++ wrapper functions for KV scoring and pruning.
struct TriAttentionState {
    // C library stats pointer (owned, freed on destruction)
    tria_stats* stats_ptr = nullptr;
    bool enabled = false;

    // Configuration (from environment or defaults)
    int kv_budget = 512;           // Max tokens to retain (default 512 for better trigger)
    int divide_length = 128;        // Compression trigger interval
    int window_size = 128;          // Recent tokens always preserved
    float min_keep_ratio = 0.5f;    // Minimum keep ratio (default 50% to prevent over-compression)
    bool force_compress = false;    // Force compression regardless of budget (for testing)

    // Runtime state
    int last_compressed_pos = 0;    // Track when we last compressed

    // Constructor/destructor
    TriAttentionState() = default;
    ~TriAttentionState() {
        if (stats_ptr) {
            tria_free(stats_ptr);
            stats_ptr = nullptr;
        }
    }

    // Disable copy, enable move
    TriAttentionState(const TriAttentionState&) = delete;
    TriAttentionState& operator=(const TriAttentionState&) = delete;
    TriAttentionState(TriAttentionState&& other) noexcept
        : stats_ptr(other.stats_ptr), enabled(other.enabled),
          kv_budget(other.kv_budget), divide_length(other.divide_length),
          window_size(other.window_size), min_keep_ratio(other.min_keep_ratio),
          force_compress(other.force_compress), last_compressed_pos(other.last_compressed_pos) {
        other.stats_ptr = nullptr;
    }

    // Load stats from a .bin file. Returns false on error.
    // The stats file must match the model architecture (layers, heads, head_dim).
    bool load_stats(const std::string& path) {
        stats_ptr = tria_load(path.c_str());
        enabled = (stats_ptr != nullptr);
        return enabled;
    }

    // Check if compression should trigger at the given position.
    // If force_compress is set, only the divide_length check applies.
    // Otherwise, only compress when (1) divide_length has passed AND (2) committed tokens
    // exceed kv_budget. Avoids compressing when cache is already small.
    bool should_compress(int cur_pos, int committed) const {
        if (!enabled) return false;
        if (cur_pos - last_compressed_pos < divide_length) return false;
        if (force_compress) return true;  // Bypass budget check (testing/debug mode)
        if (committed <= kv_budget) return false;  // No need to compress when under budget
        return true;
    }

    // Mark compression as done at this position.
    void mark_compressed(int cur_pos) {
        last_compressed_pos = cur_pos;
    }

    // Get per-layer budget for TriAttention.
    // Returns: floor(kv_budget * layer_budget_scale)
    int layer_budget(int layer_idx, const tria_stats* stats) const {
        if (!stats) return kv_budget;
        float scale = stats->layer_budget_scales[layer_idx];
        int b = static_cast<int>(kv_budget * scale);
        return (b > 0) ? b : 1;
    }
};

// Global TriAttention state (initialized at startup, freed at shutdown)
// This is a simple singleton for the initial integration.
// In a full implementation, this would be part of TargetCache.
extern TriAttentionState g_tria_state;

// Initialize TriAttention from environment variables.
// Reads:
//   - TRIATTN_STATS_PATH: path to .bin stats file
//   - TRIATTN_KV_BUDGET: max tokens to retain (default 512)
//   - TRIATTN_DIVIDE_LENGTH: compression interval (default 128)
//   - TRIATTN_WINDOW_SIZE: recent tokens preserved (default 128)
//   - TRIATTN_MIN_KEEP_RATIO: minimum keep ratio to prevent over-compression (default 0.5)
//   - TRIATTN_FORCE_COMPRESS: force compression every divide_length regardless of budget (default 0)
void init_triattention_from_env();

// Free TriAttention resources
void free_triattention();

// TriAttention KV compression — reads pre-RoPE K from GPU, scores per head,
// selects top-K positions, compacts KV cache in-place.
// Gated by TRIATTN_ENABLED=1 env var (via stats file presence).
//
// Parameters:
//   stats           — loaded tria_stats (from g_tria_state)
//   k_pre_rope_gpu  — GPU bf16 buffer [head_dim, max_ctx, n_head_kv]
//   attn_k          — array of K cache tensor ptrs (one per FA layer)
//   attn_v          — array of V cache tensor ptrs (one per FA layer)
//   n_full_attn     — number of full-attention layers
//   n_head_kv       — KV heads per layer
//   head_dim        — per-head dimension
//   max_ctx         — allocated max context
//   kv_start        — start of the KV range to consider compressing (0 = from beginning)
//   cur_pos         — current context length (positions 0..cur_pos are populated)
//   gpu_id          — GPU device ID for D2H/H2D copies
//   keep_ratio      — fraction of positions to keep (0.0-1.0)
//   n_kept_out      — [out] number of positions kept after compression
//   k_type          — ggml_type of K cache (for element size calculation)
//   v_type          — ggml_type of V cache (for element size calculation)
//
// Returns true on success. The KV cache is compacted in-place: kept positions
// are moved to the front [kv_start..kv_start+n_kept). cur_pos should be updated
// to kv_start + n_kept by the caller.
//
// head_dim: TriAttention RoPE head_dim (from stats, e.g., 64)
// tensor_head_dim: Actual K tensor head_dim for buffer layout (e.g., 256)
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
    enum ggml_type v_type);

} // namespace dflash27b
