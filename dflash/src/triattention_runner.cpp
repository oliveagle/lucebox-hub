// triattention_runner.cpp — TriAttention runtime implementation for DFlash
//
// This file implements the C++ wrapper for TriAttention KV compression.
// It provides initialization from environment variables and the compression
// trigger logic.

#include "triattention_runner.h"

#if defined(DFLASH27B_TRIATTENTION_ENABLED)
#include "triattention.h"
#endif

#include <cstdlib>
#include <cstring>
#include <cstdio>

namespace dflash27b {

// Global TriAttention state
TriAttentionState g_tria_state;

void init_triattention_from_env() {
#if defined(DFLASH27B_TRIATTENTION_ENABLED)
    // Read stats path from environment
    const char* stats_path = std::getenv("TRIATTN_STATS_PATH");
    if (!stats_path) {
        // Default path for Qwen3.5-27B stats
        stats_path = "dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin";
    }

    // Check if TriAttention is enabled via env var
    const char* tria_enabled = std::getenv("TRIATTN_ENABLED");
    if (tria_enabled && std::strcmp(tria_enabled, "1") != 0) {
        std::fprintf(stderr, "[TriAttention] Disabled by TRIATTN_ENABLED=%s\n", tria_enabled);
        return;
    }

    // Load stats using the C library loader
    g_tria_state.stats_ptr = tria_load(stats_path);
    if (!g_tria_state.stats_ptr) {
        std::fprintf(stderr, "[TriAttention] Failed to load stats from: %s\n", stats_path);
        return;
    }

    g_tria_state.enabled = true;
    std::fprintf(stderr, "[TriAttention] Loaded stats with %u layers, %u heads, head_dim=%u\n",
                g_tria_state.stats_ptr->num_layers,
                g_tria_state.stats_ptr->num_heads,
                g_tria_state.stats_ptr->head_dim);

    // Read configuration from environment
    const char* kv_budget_env = std::getenv("TRIATTN_KV_BUDGET");
    if (kv_budget_env) {
        g_tria_state.kv_budget = std::atoi(kv_budget_env);
    }

    const char* divide_length_env = std::getenv("TRIATTN_DIVIDE_LENGTH");
    if (divide_length_env) {
        g_tria_state.divide_length = std::atoi(divide_length_env);
    }

    const char* window_size_env = std::getenv("TRIATTN_WINDOW_SIZE");
    if (window_size_env) {
        g_tria_state.window_size = std::atoi(window_size_env);
    }

    std::fprintf(stderr, "[TriAttention] Enabled: kv_budget=%d, divide_length=%d, window_size=%d\n",
                g_tria_state.kv_budget, g_tria_state.divide_length, g_tria_state.window_size);
#else
    std::fprintf(stderr, "[TriAttention] TriAttention not compiled in (DFLASH27B_TRIATTENTION_ENABLED not set)\n");
#endif
}

void free_triattention() {
#if defined(DFLASH27B_TRIATTENTION_ENABLED)
    if (g_tria_state.stats_ptr) {
        tria_free(g_tria_state.stats_ptr);
        g_tria_state.stats_ptr = nullptr;
    }
    g_tria_state.enabled = false;
#endif
}

} // namespace dflash27b