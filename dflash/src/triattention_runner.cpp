// triattention_runner.cpp — TriAttention runtime implementation for DFlash
//
// This file implements the C++ wrapper for TriAttention KV compression.
//
// INTEGRATION STATUS (2026-05-18):
//   Phase 1 (COMPLETE): C library wrapper and environment initialization
//   Phase 2 (TODO): Pre-RoPE key capture in compute graph
//   Phase 3 (TODO): KV scoring and pruning in decode loop

#include "triattention_runner.h"
#include <cstdlib>
#include <cstring>

namespace dflash27b {

// Global TriAttention state
TriAttentionState g_tria_state;

void init_triattention_from_env() {
    // Read stats path from environment
    const char* stats_path = std::getenv("TRIATTN_STATS_PATH");
    if (!stats_path) {
        // Default path for Qwen3.5-27B stats
        stats_path = "dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin";
    }

    // Load stats
    if (!g_tria_state.load_stats(stats_path)) {
        // Stats loading failed - TriAttention will be disabled
        return;
    }

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
}

} // namespace dflash27b

// NOTE: The actual KV scoring and pruning would be implemented in
// qwen35_target_graph.cpp::build_full_attn_block() after the KV write.
//
// Pseudo-code for the integration:
//
//   // In build_full_attn_block(), after Kcur/Vcur projection but before RoPE:
//   if (g_tria_state.enabled && g_tria_state.should_compress(kv_start + n_tokens)) {
//       // Capture pre-RoPE K (need to store this somewhere accessible)
//       // This requires modifying the compute graph to retain pre-RoPE K
//
//       // For each KV head, call tria_score_kv_head()
//       std::vector<float> scores(seq_len * n_head_kv);
//       for (int h = 0; h < n_head_kv; h++) {
//           tria_score_kv_head(
//               g_tria_state.stats_ptr,
//               k_pre_real, k_pre_imag,  // Pre-RoPE K for this head
//               key_positions,           // Position IDs
//               kv_start + n_tokens,     // Current position
//               seq_len,                 // Sequence length
//               layer_idx,               // Layer index
//               h,                       // KV head index
//               scores.data() + h * seq_len  // Output scores
//           );
//       }
//
//       // Aggregate scores across GQA groups and select top-K
//       int budget = g_tria_state.layer_budget(layer_idx, g_tria_state.stats_ptr);
//       std::vector<int> keep_indices = select_top_k(scores, budget);
//
//       // Compact KV cache to keep only selected indices
//       compact_kv_cache(cache_k, cache_v, keep_indices);
//
//       g_tria_state.mark_compressed(kv_start + n_tokens);
//   }
//
// CHALLENGES:
//   1. Pre-RoPE K capture: DFlash applies RoPE inline, so we need to
//      either (a) store pre-RoPE K separately, or (b) split the RoPE op
//   2. GPU-CPU sync: Scoring happens on CPU, KV is on GPU
//   3. KV compaction: Need to rewrite the KV cache in-place
//   4. Position mapping: After compaction, position IDs need remapping
//
// RECOMMENDED APPROACH:
//   For now, use vLLM + TriAttention for the KV compression path,
//   and keep DFlash focused on speculative decoding without KV compression.
//   The two can be compared in benchmarks to quantify the trade-offs.
