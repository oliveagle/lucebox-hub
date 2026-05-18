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

// Forward declaration of the C struct
struct tria_stats;

// C linkage for the TriAttention C library functions
extern "C" {
    struct tria_stats * tria_load(const char *path);
    void tria_free(struct tria_stats *stats);
    void tria_score_kv_head(
        const struct tria_stats *stats,
        const float *k_pre_real,
        const float *k_pre_imag,
        const int   *key_pos,
        int          cur_pos,
        int          seq_len,
        int          layer_idx,
        int          kv_head_idx,
        float       *out_scores
    );
}

namespace dflash27b {

// TriAttention runtime state.
// Manages the lifecycle of the tria_stats pointer and provides
// C++ wrapper functions for KV scoring and pruning.
struct TriAttentionState {
    // C library stats pointer (owned, freed on destruction)
    tria_stats* stats_ptr = nullptr;
    bool enabled = false;

    // Configuration (from environment or defaults)
    int kv_budget = 2048;           // Max tokens to retain
    int divide_length = 128;        // Compression trigger interval
    int window_size = 128;          // Recent tokens always preserved

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
          window_size(other.window_size), last_compressed_pos(other.last_compressed_pos) {
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
    // Returns true if (cur_pos - last_compressed_pos) >= divide_length.
    bool should_compress(int cur_pos) const {
        if (!enabled) return false;
        return (cur_pos - last_compressed_pos) >= divide_length;
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
//   - TRIATTN_KV_BUDGET: max tokens to retain (default 2048)
//   - TRIATTN_DIVIDE_LENGTH: compression interval (default 128)
//   - TRIATTN_WINDOW_SIZE: recent tokens preserved (default 128)
void init_triattention_from_env();

} // namespace dflash27b
