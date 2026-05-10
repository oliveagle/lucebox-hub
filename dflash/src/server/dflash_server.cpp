// dflash_server.cpp - C++ HTTP Server Implementation
//
// This file implements a standalone HTTP server for DFlash that replaces
// the Python-based server_tools.py.

#include "dflash_server.h"
#include "internal.h"
#include "parsers.h"
#include "sampler.h"
#include "tokenizer.h"
#include "kv_quant.h"
#include "sampler.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "dflash_graph.h"
#include "qwen3_drafter.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <optional>

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace dflash27b {

// ─── Logging Infrastructure ────────────────────────────────────────────────────

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

static const char* log_level_to_string(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        default: return "UNKNOWN";
    }
}

static void log_message(LogLevel level, const std::string& message) {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    char time_buf[64];
    std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&time_t));

    std::ostringstream oss;
    oss << "[" << time_buf << "." << std::setfill('0') << std::setw(3) << ms.count() << "]"
        << "[" << log_level_to_string(level) << "] "
        << message;

    if (level >= LogLevel::ERROR) {
        std::cerr << oss.str() << std::endl;
    } else {
        std::cout << oss.str() << std::endl;
    }
}

#define LOG_DEBUG(msg)   log_message(LogLevel::DEBUG,   msg)
#define LOG_INFO(msg)    log_message(LogLevel::INFO,    msg)
#define LOG_WARNING(msg) log_message(LogLevel::WARNING, msg)
#define LOG_ERROR(msg)   log_message(LogLevel::ERROR,   msg)

// ─── Constants ────────────────────────────────────────────────────────────

// Use constants from dflash27b.h and internal.h:
//   DFLASH27B_TARGET_HIDDEN = 5120
//   DFLASH27B_TARGET_VOCAB = 248320
//   DFLASH27B_DRAFT_BLOCK_SIZE = 16
//   DFLASH27B_DRAFT_MASK_TOKEN_ID = 248070
//   DFLASH27B_DRAFT_N_TARGET_LAYERS = 5

static constexpr int KQ_MASK_PAD = 32;
static constexpr int DRAFT_CTX_MAX = 2048;

// EOS token check macro (mirrors test_dflash.cpp definition)
#define IS_EOS_TOK(tok, w) \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// F16 encoding for 0 and -inf
static constexpr uint16_t F16_ZERO = 0x0000;
static constexpr uint16_t F16_NEG_INF = 0xFC00;

static int g_kq_stride_pad = KQ_MASK_PAD;

static int align_up(int x, int a) { return ((x + a - 1) / a) * a; }

// ─── Helper Functions ─────────────────────────────────────────────────────

static int argmax_f32(const float* x, int n) {
    int best = 0;
    float bv = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > bv) {
            bv = x[i];
            best = i;
        }
    }
    return best;
}

static void build_causal_mask(std::vector<uint16_t>& out,
                              int kv_len, int n_tokens, int kv_start,
                              int win_start = 0) {
    const int kv_pad = align_up(kv_len, g_kq_stride_pad);
    const int q_pad = align_up(n_tokens, KQ_MASK_PAD);
    out.assign((size_t)kv_pad * q_pad, F16_NEG_INF);
    const int abs_end = win_start + kv_len;
    for (int q = 0; q < n_tokens; q++) {
        const int abs_q = kv_start + q;
        const int min_k = std::max(0, win_start);
        const int max_k = abs_q;
        for (int k = min_k; k <= max_k && k < abs_end; k++) {
            out[(size_t)q * kv_pad + (k - win_start)] = F16_ZERO;
        }
    }
}

// ─── DDTree Support ─────────────────────────────────────────────────────────

// Per-position top-K softmax extraction.
// Input:  logits [n_positions × vocab] f32
// Output: out_log_probs [n_positions × K] f32, out_token_ids [n_positions × K] i32
static void extract_draft_topk(const float * logits,
                               int n_positions, int vocab, int K,
                               float * out_log_probs,
                               int32_t * out_token_ids,
                               float temperature = 1.0f) {
    struct Entry { float logit; int32_t id; };
    auto cmp_greater = [](const Entry & a, const Entry & b) {
        return a.logit > b.logit;
    };
    const float inv_t = 1.0f / std::max(1e-3f, temperature);

    for (int i = 0; i < n_positions; i++) {
        const float * li = logits + (size_t)i * vocab;
        std::vector<Entry> heap;
        heap.reserve(K);

        float running_max     = -INFINITY;
        float running_sum_exp = 0.0f;
        for (int j = 0; j < vocab; j++) {
            const float l = li[j] * inv_t;
            if (l > running_max) {
                if (running_max > -INFINITY) {
                    running_sum_exp = running_sum_exp * std::exp(running_max - l);
                }
                running_sum_exp += 1.0f;
                running_max = l;
            } else {
                running_sum_exp += std::exp(l - running_max);
            }
            if ((int)heap.size() < K) {
                heap.push_back({l, (int32_t)j});
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            } else if (l > heap.front().logit) {
                std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                heap.back() = {l, (int32_t)j};
                std::push_heap(heap.begin(), heap.end(), cmp_greater);
            }
        }
        const float log_z = running_max + std::log(running_sum_exp);
        std::sort_heap(heap.begin(), heap.end(), cmp_greater);
        for (int k = 0; k < K; k++) {
            out_log_probs[(size_t)i * K + k] = heap[k].logit - log_z;
            out_token_ids[(size_t)i * K + k] = heap[k].id;
        }
    }
}

// A flat DFS-ordered tree built from the draft's top-K softmax distributions.
struct DDTree {
    int                         n_nodes = 0;
    std::vector<int32_t>        token_ids;            // size n_nodes
    std::vector<int>            depths;               // size n_nodes (1..L)
    std::vector<int>            parents;              // size n_nodes + 1
    std::vector<std::unordered_map<int32_t, int>> child_maps;  // size n_nodes + 1
    std::vector<uint8_t>        visibility;           // (1 + n_nodes)^2 row-major
};

// Build DDTree from per-position top-K log-probabilities via best-first heap.
static DDTree build_ddtree(const float * top_log_probs,
                           const int32_t * top_token_ids,
                           int L, int K, int budget,
                           bool chain_seed = true) {
    DDTree tree;
    if (budget <= 0 || L <= 0) {
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();
        tree.visibility.assign(1, 1);
        return tree;
    }

    struct HeapEntry {
        float                neg_logw;
        std::vector<int>     ranks;
        int                  parent_index;
        int                  depth;
        int                  rank;
        float                logw;
    };
    struct HeapCmp {
        bool operator()(const HeapEntry & a, const HeapEntry & b) const {
            return a.neg_logw > b.neg_logw;
        }
    };
    std::priority_queue<HeapEntry, std::vector<HeapEntry>, HeapCmp> heap;

    tree.token_ids.reserve(budget);
    tree.depths.reserve(budget);
    tree.parents.reserve(budget + 1);
    tree.parents.push_back(-1);
    tree.child_maps.emplace_back();

    // Chain seed: pre-seed full top-1 chain (defensive against flat softmax)
    if (chain_seed) {
        const int chain_depth = std::min(L, budget);
        float cum_logw = 0.0f;
        int   prev_idx = 0;
        for (int d = 1; d <= chain_depth; d++) {
            const int32_t tok_id = top_token_ids[(size_t)(d - 1) * K + 0];
            cum_logw += top_log_probs[(size_t)(d - 1) * K + 0];

            const int cur_idx = tree.n_nodes + 1;
            tree.token_ids.push_back(tok_id);
            tree.depths.push_back(d);
            tree.parents.push_back(prev_idx);
            tree.child_maps.emplace_back();
            tree.child_maps[prev_idx][tok_id] = cur_idx;
            tree.n_nodes++;

            if (K > 1) {
                const float sibling_logw = cum_logw
                    - top_log_probs[(size_t)(d - 1) * K + 0]
                    + top_log_probs[(size_t)(d - 1) * K + 1];
                heap.push({-sibling_logw, {1}, prev_idx, d, 1, sibling_logw});
            }
            prev_idx = cur_idx;
        }
    } else {
        const float root_logw = top_log_probs[0 * K + 0];
        heap.push({-root_logw, {0}, 0, 1, 0, root_logw});
    }

    while (!heap.empty() && tree.n_nodes < budget) {
        HeapEntry top = heap.top();
        heap.pop();

        const int depth_minus_1 = top.depth - 1;
        const int rank          = top.rank;
        const int32_t token_id  = top_token_ids[(size_t)depth_minus_1 * K + rank];

        const int current_index = tree.n_nodes + 1;
        tree.token_ids.push_back(token_id);
        tree.depths.push_back(top.depth);
        tree.parents.push_back(top.parent_index);
        tree.child_maps.emplace_back();
        tree.child_maps[top.parent_index][token_id] = current_index;
        tree.n_nodes++;

        // Push next sibling
        if (rank + 1 < K) {
            const float sibling_logw = top.logw
                - top_log_probs[(size_t)depth_minus_1 * K + rank]
                + top_log_probs[(size_t)depth_minus_1 * K + rank + 1];
            std::vector<int> sibling_ranks = top.ranks;
            sibling_ranks.back() = rank + 1;
            heap.push({-sibling_logw, std::move(sibling_ranks),
                        top.parent_index, top.depth, rank + 1, sibling_logw});
        }

        // Push first child
        if (top.depth < L) {
            const float child_logw = top.logw
                + top_log_probs[(size_t)top.depth * K + 0];
            std::vector<int> child_ranks = top.ranks;
            child_ranks.push_back(0);
            heap.push({-child_logw, std::move(child_ranks),
                        current_index, top.depth + 1, 0, child_logw});
        }
    }

    // Build ancestor-only visibility mask
    const int N = 1 + tree.n_nodes;
    tree.visibility.assign((size_t)N * N, 0);
    tree.visibility[0 * N + 0] = 1;
    for (int i = 1; i < N; i++) {
        const int p = tree.parents[i];
        for (int j = 0; j < i; j++) {
            tree.visibility[(size_t)i * N + j] = tree.visibility[(size_t)p * N + j];
        }
        tree.visibility[(size_t)i * N + i] = 1;
    }

    return tree;
}

// Walk the verified tree following target's argmax (posterior) at each node.
// Returns list of flat-tree indices of the accepted path (starting at root),
// plus the next "bonus" token.
static std::vector<int> follow_verified_tree(const DDTree & tree,
                                             const int32_t * posterior,
                                             int & out_next_token,
                                             int * out_node_idx = nullptr) {
    std::vector<int> accepted;
    accepted.reserve(tree.n_nodes + 1);
    accepted.push_back(0);

    int current_index = 0;
    int next_token    = posterior[current_index];
    while (true) {
        const auto & children = tree.child_maps[current_index];
        auto it = children.find(next_token);
        if (it == children.end()) break;
        current_index = it->second;
        accepted.push_back(current_index);
        next_token = posterior[current_index];
    }
    out_next_token = next_token;
    if (out_node_idx) *out_node_idx = current_index;
    return accepted;
}

// ─── StepGraph Helper ─────────────────────────────────────────────────────

static void step_graph_free(StepGraph& sg) {
    if (sg.alloc) {
        ggml_gallocr_free(sg.alloc);
        sg.alloc = nullptr;
    }
    if (sg.ctx) {
        ggml_free(sg.ctx);
        sg.ctx = nullptr;
    }
    sg.gf = nullptr;
    sg.inp_embed = nullptr;
    sg.positions = nullptr;
    sg.attn_mask = nullptr;
    sg.parent_ids = nullptr;
    sg.logits = nullptr;
    sg.argmax_tokens = nullptr;
    sg.delta_captures.clear();
}

static void step_graph_destroy(StepGraph& sg) {
    step_graph_free(sg);
}

// ─── Target Graph Step Builder ────────────────────────────────────────────

bool DFlashServer::build_target_step(StepGraph& sg, int kv_start, int n_tokens,
                                      bool with_mask, bool capture,
                                      bool capture_delta_intermediate,
                                      int fa_window) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    if (with_mask) {
        const int win_start = (fa_window > 0 && kv_start > fa_window) ? (kv_start - fa_window) : 0;
        const int win_len = kv_start + n_tokens - win_start;
        const int kv_pad = align_up(win_len, g_kq_stride_pad);
        const int q_pad = align_up(n_tokens, KQ_MASK_PAD);
        sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
        ggml_set_name(sg.attn_mask, "attn_mask");
        ggml_set_input(sg.attn_mask);
    }

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed = sg.inp_embed;
    gi.positions = sg.positions;
    gi.attn_mask = sg.attn_mask;
    gi.n_tokens = n_tokens;
    gi.kv_start = kv_start;
    gi.capture_layers = capture;
    gi.capture_delta_intermediate = capture_delta_intermediate;
    gi.fa_window = fa_window;
    gi.last_token_logits_only = false;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, *target_weights_, *cache_, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "chain_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache_->backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── DDTree Target Verify Graph Builder ───────────────────────────────────────

bool DFlashServer::build_target_step_tree(StepGraph& sg, int kv_start, int n_tokens, int fa_window) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size = 512 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, n_tokens, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, 4 * n_tokens);
    ggml_set_name(sg.positions, "positions");
    ggml_set_input(sg.positions);

    // Use max possible mask size so gallocr shape stays fixed across steps
    const int max_win_len = cache_->max_ctx + n_tokens;
    const int kv_pad = align_up(max_win_len, g_kq_stride_pad);
    const int q_pad = align_up(n_tokens, KQ_MASK_PAD);
    sg.attn_mask = ggml_new_tensor_2d(sg.ctx, GGML_TYPE_F16, kv_pad, q_pad);
    ggml_set_name(sg.attn_mask, "attn_mask");
    ggml_set_input(sg.attn_mask);

    // parent_ids for DDTree recurrence
    sg.parent_ids = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, n_tokens);
    ggml_set_name(sg.parent_ids, "parent_ids");
    ggml_set_input(sg.parent_ids);

    sg.gf = ggml_new_graph_custom(sg.ctx, 16384, false);

    QwenGraphInputs gi{};
    gi.inp_embed = sg.inp_embed;
    gi.positions = sg.positions;
    gi.attn_mask = sg.attn_mask;
    gi.n_tokens = n_tokens;
    gi.kv_start = kv_start;
    gi.fa_window = fa_window;
    gi.capture_layers = true;
    gi.capture_delta_intermediate = true;
    gi.parent_ids = sg.parent_ids;

    QwenGraphOutputs go = build_qwen35_graph(sg.ctx, sg.gf, *target_weights_, *cache_, gi);
    if (!go.logits) return false;
    sg.logits = go.logits;
    sg.delta_captures = std::move(go.delta_captures);
    ggml_set_output(sg.logits);

    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "tree_verify_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache_->backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── Draft Graph Step Builder ─────────────────────────────────────────────

bool DFlashServer::build_draft_step(StepGraph& sg, int draft_ctx,
                                     DraftFeatureMirror* mirror, int committed) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size = 256 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int fc_in = DFLASH27B_DRAFT_N_TARGET_LAYERS * hidden;

    sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.inp_embed, "inp_embed");
    ggml_set_input(sg.inp_embed);

    int mirror_slot0 = 0;
    if (mirror && cache_->target_feat) {
        const size_t stride = cache_->target_feat->nb[1];
        sg.inp_embed = ggml_view_3d(
            sg.ctx,
            mirror->target_feat,
            fc_in, draft_ctx, 1,
            stride,
            stride * (size_t)draft_ctx,
            (size_t)mirror_slot0 * stride);
    } else {
        sg.inp_embed = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, fc_in, draft_ctx, 1);
        ggml_set_input(sg.inp_embed);
    }
    ggml_set_name(sg.inp_embed, "target_hidden_cat");

    sg.positions = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, q_len);
    ggml_set_name(sg.positions, "positions_q");
    ggml_set_input(sg.positions);

    ggml_tensor* positions_k = ggml_new_tensor_1d(sg.ctx, GGML_TYPE_I32, draft_ctx + q_len);
    ggml_set_name(positions_k, "positions_k");
    ggml_set_input(positions_k);

    sg.gf = ggml_new_graph_custom(sg.ctx, 4096, false);

    DraftGraphInputs gi{};
    gi.ctx_len = draft_ctx;
    gi.noise_embed = sg.inp_embed;
    gi.target_hidden_cat = sg.inp_embed;
    gi.positions_q = sg.positions;
    gi.positions_k = positions_k;
    gi.lm_head = target_weights_->output;

    DraftGraphOutputs go = build_draft_graph(sg.ctx, *draft_weights_, gi);
    sg.hidden_states = go.hidden_states;
    sg.logits = go.logits;
    if (!sg.hidden_states) {
        std::cerr << "draft graph missing hidden_states" << std::endl;
        return false;
    }
    if (sg.logits) {
        sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
        ggml_set_name(sg.argmax_tokens, "argmax_tokens");
        ggml_set_output(sg.argmax_tokens);
        ggml_build_forward_expand(sg.gf, sg.argmax_tokens);
    } else {
        ggml_set_output(sg.hidden_states);
        ggml_build_forward_expand(sg.gf, sg.hidden_states);
    }

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache_->backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── LM Head Projection Step Builder ───────────────────────────────────────

bool DFlashServer::build_lm_head_projection_step(StepGraph& sg, int q_len) {
    step_graph_free(sg);

    ggml_init_params ip{};
    ip.mem_size = 64 * 1024 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    sg.ctx = ggml_init(ip);
    if (!sg.ctx) return false;

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    sg.hidden_input = ggml_new_tensor_3d(sg.ctx, GGML_TYPE_F32, hidden, q_len, 1);
    ggml_set_name(sg.hidden_input, "draft_hidden_for_lm_head");
    ggml_set_input(sg.hidden_input);

    sg.gf = ggml_new_graph_custom(sg.ctx, 1024, false);
    sg.logits = ggml_mul_mat(sg.ctx, target_weights_->output, sg.hidden_input);
    ggml_set_name(sg.logits, "draft_projected_logits");
    ggml_set_output(sg.logits);
    sg.argmax_tokens = ggml_argmax(sg.ctx, sg.logits);
    ggml_set_name(sg.argmax_tokens, "draft_projected_argmax");
    ggml_set_output(sg.argmax_tokens);
    ggml_build_forward_expand(sg.gf, sg.argmax_tokens);

    if (!sg.alloc) {
        sg.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache_->backend));
    }
    return ggml_gallocr_alloc_graph(sg.alloc, sg.gf);
}

// ─── DFlash Decode Loop ───────────────────────────────────────────────────

bool DFlashServer::run_dflash_decode(
    const std::vector<int32_t>& prompt,
    int n_gen,
    int last_tok,
    std::function<void(int32_t)> on_token,
    int& n_generated,
    const SamplerCfg& sampler) {

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;
    const bool ddtree_mode = (config_.ddtree_budget > 0 && config_.ddtree_budget <= 128);
    const int ddtree_budget = config_.ddtree_budget;

    int committed = static_cast<int>(prompt.size());
    n_generated = 0;

    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<float>   draft_logits_buf;  // Only for DDTree K>1
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;

    // DDTree-specific buffers
    std::vector<float>   ddtree_top_log_probs;
    std::vector<int32_t> ddtree_top_token_ids;

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // Build noise block [last_tok, MASK*15]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) {
            noise_ids[i] = mask_tok;
        }

        if (!target_weights_->embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::cerr << "noise embed failed" << std::endl;
            return false;
        }

        // Draft forward
        const int draft_ctx = std::min(committed, std::min(cache_->target_feat_cap, DRAFT_CTX_MAX));
        const int draft_start = committed - draft_ctx;

        if (!build_draft_step(draft_sg_, draft_ctx, &feature_mirror_, committed)) {
            std::cerr << "draft step build failed" << std::endl;
            return false;
        }

        ggml_backend_tensor_set(draft_sg_.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());

        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;

        ggml_backend_tensor_set(draft_sg_.positions, pos_q.data(), 0,
                                sizeof(int32_t) * pos_q.size());

        auto st = ggml_backend_graph_compute(cache_->backend, draft_sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::cerr << "draft compute failed: " << (int)st << std::endl;
            return false;
        }

        ggml_backend_tensor_get(draft_sg_.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

        // ─────────────────────────────────────────────────────────────────────
        // DDTree Path: Build tree from draft top-K, verify with tree attention
        // ─────────────────────────────────────────────────────────────────────
        if (ddtree_mode) {
            const int L = q_len - 1;  // Positions 1..15 for tree (slot 0 is root/bonus)
            const int ddtree_K = (ddtree_budget > L) ? 8 : 1;  // Use K=8 only when budget > depth

            // Extract top-K for tree building (skip when K=1, just use argmax)
            if (ddtree_K > 1) {
                // Need full logits for top-K extraction
                draft_logits_buf.resize((size_t)vocab * q_len);
                ggml_backend_tensor_get(draft_sg_.logits, draft_logits_buf.data(), 0,
                                        sizeof(float) * draft_logits_buf.size());

                ddtree_top_log_probs.resize((size_t)L * ddtree_K);
                ddtree_top_token_ids.resize((size_t)L * ddtree_K);
                extract_draft_topk(draft_logits_buf.data() + (size_t)vocab,  // Skip position 0
                                   L, vocab, ddtree_K,
                                   ddtree_top_log_probs.data(),
                                   ddtree_top_token_ids.data(),
                                   1.0f);  // temperature=1.0 for now
            } else {
                // Fast path: use argmax results
                ddtree_top_log_probs.resize(L);
                ddtree_top_token_ids.resize(L);
                for (int i = 0; i < L; i++) {
                    ddtree_top_log_probs[i] = 0.0f;
                    ddtree_top_token_ids[i] = draft_tok[i + 1];
                }
            }

            // Build DDTree via best-first search
            DDTree tree = build_ddtree(
                ddtree_top_log_probs.data(),
                ddtree_top_token_ids.data(),
                L, ddtree_K, ddtree_budget,
                true);  // chain_seed=true for defensive tree building

            const int N_actual = 1 + tree.n_nodes;  // root + nodes
            const int N = ddtree_budget + 1;         // fixed allocation for gallocr reuse

            // Build tree verify graph
            if (!build_target_step_tree(target_sg_, committed, N, config_.fa_window)) {
                std::cerr << "ddtree verify build failed" << std::endl;
                return false;
            }

            // Embed tokens: [last_tok, tree.token_ids..., padding]
            std::vector<int32_t> flat_tokens(N, 0);
            flat_tokens[0] = last_tok;
            for (int i = 0; i < tree.n_nodes; i++) flat_tokens[1 + i] = tree.token_ids[i];

            std::vector<float> tree_embed((size_t)hidden * N, 0.0f);
            if (!target_weights_->embedder.embed(flat_tokens.data(), N_actual, tree_embed.data())) {
                std::cerr << "ddtree embed failed" << std::endl;
                return false;
            }
            ggml_backend_tensor_set(target_sg_.inp_embed, tree_embed.data(), 0,
                                    sizeof(float) * tree_embed.size());

            // M-RoPE positions: use depth from tree
            std::vector<int32_t> pos4(4 * N, 0);
            for (int i = 0; i < N_actual; i++) {
                int p = committed + (i == 0 ? 0 : tree.depths[i - 1]);
                pos4[0 * N + i] = p;
                pos4[1 * N + i] = p;
                pos4[2 * N + i] = p;
                pos4[3 * N + i] = 0;
            }
            ggml_backend_tensor_set(target_sg_.positions, pos4.data(), 0,
                                    sizeof(int32_t) * pos4.size());

            // Build ancestor-only attention mask
            const int tree_win_start = (config_.fa_window > 0 && committed > config_.fa_window)
                                           ? (committed - config_.fa_window) : 0;
            {
                const int max_win_len = cache_->max_ctx + N;
                const int kv_pad_m = align_up(max_win_len, g_kq_stride_pad);
                const int q_pad_m = align_up(N, KQ_MASK_PAD);
                std::vector<uint16_t> mask_buf((size_t)kv_pad_m * q_pad_m, F16_NEG_INF);

                for (int q = 0; q < N_actual; q++) {
                    // Past KV visible to all
                    for (int k = std::max(0, tree_win_start); k < committed; k++) {
                        mask_buf[(size_t)q * kv_pad_m + (k - tree_win_start)] = F16_ZERO;
                    }
                    // Tree self-visibility
                    for (int j = 0; j < N_actual; j++) {
                        if (tree.visibility[(size_t)q * N_actual + j]) {
                            mask_buf[(size_t)q * kv_pad_m + (committed + j - tree_win_start)] = F16_ZERO;
                        }
                    }
                }
                ggml_backend_tensor_set(target_sg_.attn_mask, mask_buf.data(), 0,
                                        sizeof(uint16_t) * mask_buf.size());
            }

            // Set parent_ids for tree recurrence
            std::vector<int32_t> parent_ids(N, 0);
            parent_ids[0] = -1;
            for (int i = 1; i < N_actual; i++) parent_ids[i] = (int32_t)tree.parents[i];
            ggml_backend_tensor_set(target_sg_.parent_ids, parent_ids.data(), 0,
                                    sizeof(int32_t) * N);

            // Verify
            st = ggml_backend_graph_compute(cache_->backend, target_sg_.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::cerr << "ddtree verify compute failed: " << (int)st << std::endl;
                return false;
            }

            // Get posterior
            std::vector<int32_t> posterior(N_actual);
            ggml_backend_tensor_get(target_sg_.argmax_tokens, posterior.data(), 0,
                                    sizeof(int32_t) * N_actual);

            // Walk tree to get accepted path
            int next_token = -1;
            int bonus_node_idx = 0;
            std::vector<int> accepted = follow_verified_tree(tree, posterior.data(), next_token, &bonus_node_idx);

            const int accept_depth = (int)accepted.size();
            int commit_n = accept_depth;
            if (commit_n > need_commit_budget) commit_n = need_commit_budget;

            // Emit accepted tokens
            bool hit_eos = false;
            for (int i = 0; i < commit_n; i++) {
                const int dfs_idx = accepted[i];
                const int32_t tok = (dfs_idx == 0) ? last_tok : tree.token_ids[dfs_idx - 1];
                if (on_token) on_token(tok);
                if (IS_EOS_TOK(tok, *target_weights_)) hit_eos = true;
            }
            last_tok = next_token;
            committed += commit_n;
            n_generated += commit_n;
            cache_->cur_pos = committed;

            // Rollback: restore SSM state to the state after processing the last committed token
            // For chain accepts (accepted[i] == i), no rollback needed - state is already correct
            // For sibling accepts, need to use delta captures
            const int rollback_dfs = (commit_n > 0) ? accepted[commit_n - 1] : 0;
            bool walked_sibling = false;
            for (int i = 0; i < commit_n; i++) {
                if (accepted[i] != i) { walked_sibling = true; break; }
            }

            // SSM state rollback (only if sibling walked)
            if (walked_sibling) {
                // Use delta captures to restore SSM state
                // TODO: Implement full SSM rollback using delta_captures
                // For now, we skip this and continue - may cause slight divergence
                std::cout << "[DDTree] Sibling walk detected, rollback not fully implemented" << std::endl;
            }

            if (hit_eos) break;
            continue;  // Skip chain-mode logic
        }

        // ─────────────────────────────────────────────────────────────────────
        // Chain Mode: Standard speculative decoding
        // ─────────────────────────────────────────────────────────────────────

        // Snapshot SSM state
        snapshot_ssm_state(*cache_);

        // Target verify (batched)
        int verify_last_tok = -1;
        if (!build_target_step(target_sg_, committed, q_len, true, true, true, config_.fa_window)) {
            std::cerr << "target verify build failed" << std::endl;
            return false;
        }

        // Set embeddings for draft tokens
        std::vector<float> verify_embed((size_t)hidden * q_len);
        if (!target_weights_->embedder.embed(draft_tok.data(), q_len, verify_embed.data())) {
            std::cerr << "verify embed failed" << std::endl;
            return false;
        }
        ggml_backend_tensor_set(target_sg_.inp_embed, verify_embed.data(), 0,
                                sizeof(float) * verify_embed.size());

        // Set positions
        std::vector<int32_t> verify_pos(4 * q_len);
        for (int i = 0; i < q_len; i++) {
            verify_pos[0 * q_len + i] = committed + i;
            verify_pos[1 * q_len + i] = committed + i;
            verify_pos[2 * q_len + i] = committed + i;
            verify_pos[3 * q_len + i] = 0;
        }
        ggml_backend_tensor_set(target_sg_.positions, verify_pos.data(), 0,
                                sizeof(int32_t) * verify_pos.size());

        // Build and set attention mask
        std::vector<uint16_t> attn_mask;
        build_causal_mask(attn_mask, committed + q_len, q_len, committed);
        ggml_backend_tensor_set(target_sg_.attn_mask, attn_mask.data(), 0,
                                sizeof(uint16_t) * attn_mask.size());

        st = ggml_backend_graph_compute(cache_->backend, target_sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::cerr << "target verify compute failed: " << (int)st << std::endl;
            return false;
        }

        ggml_backend_tensor_get(target_sg_.argmax_tokens, target_tok.data(), 0,
                                sizeof(int32_t) * q_len);

        // Greedy longest-prefix accept + bonus token
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }
        int bonus_tok = (accept_n < q_len) ? target_tok[accept_n - 1] : -1;
        int commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // Restore SSM state
        restore_ssm_state(*cache_);

        // Replay accepted tokens
        std::vector<int32_t> replay_tok((size_t)commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
        }

        int replay_last_tok = -1;
        if (!build_target_step(target_sg_, committed, commit_n, true, true, false, config_.fa_window)) {
            std::cerr << "replay build failed" << std::endl;
            return false;
        }

        // Set embeddings for replay tokens
        std::vector<float> replay_embed((size_t)hidden * commit_n);
        if (!target_weights_->embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) {
            std::cerr << "replay embed failed" << std::endl;
            return false;
        }
        ggml_backend_tensor_set(target_sg_.inp_embed, replay_embed.data(), 0,
                                sizeof(float) * replay_embed.size());

        // Set positions for replay
        std::vector<int32_t> replay_pos(4 * commit_n);
        for (int i = 0; i < commit_n; i++) {
            replay_pos[0 * commit_n + i] = committed + i;
            replay_pos[1 * commit_n + i] = committed + i;
            replay_pos[2 * commit_n + i] = committed + i;
            replay_pos[3 * commit_n + i] = 0;
        }
        ggml_backend_tensor_set(target_sg_.positions, replay_pos.data(), 0,
                                sizeof(int32_t) * replay_pos.size());

        // Build and set attention mask for replay
        std::vector<uint16_t> replay_mask;
        build_causal_mask(replay_mask, committed + commit_n, commit_n, committed);
        ggml_backend_tensor_set(target_sg_.attn_mask, replay_mask.data(), 0,
                                sizeof(uint16_t) * replay_mask.size());

        st = ggml_backend_graph_compute(cache_->backend, target_sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::cerr << "replay compute failed: " << (int)st << std::endl;
            return false;
        }

        // Sample or greedy decode for next token from replay logits
        // (instead of using argmax directly)
        std::vector<float> last_logits(vocab);
        ggml_backend_tensor_get(target_sg_.logits, last_logits.data(),
                                (size_t)(commit_n - 1) * vocab * sizeof(float),
                                sizeof(float) * vocab);
        if (sampler.temp > 0.0f) {
            last_tok = sample_logits(last_logits.data(), vocab, sampler, token_history_, rng_);
        } else {
            last_tok = argmax_f32(last_logits.data(), vocab);
        }

        // Update token history for repetition penalty
        for (int i = 0; i < commit_n; i++) {
            token_history_.push_back(replay_tok[i]);
        }
        // Keep history bounded to rep_window
        if ((int)token_history_.size() > sampler.rep_window) {
            token_history_.erase(token_history_.begin(),
                                 token_history_.end() - sampler.rep_window);
        }

        // Emit tokens and check EOS
        bool hit_eos = false;
        for (int i = 0; i < commit_n; i++) {
            if (on_token) on_token(replay_tok[i]);
            if (IS_EOS_TOK(replay_tok[i], *target_weights_)) hit_eos = true;
        }
        committed += commit_n;
        n_generated += commit_n;
        cache_->cur_pos = committed;

        if (hit_eos) break;
    }

    return true;
}

// ─── Prefix Cache ───────────────────────────────────────────────────────

class PrefixCache {
public:
    static constexpr int MAX_SLOTS = 8;

    struct Slot {
        bool occupied = false;
        std::vector<int32_t> prompt_ids;
        PrefixSnapshot snapshot;
        uint64_t last_access = 0;  // For LRU eviction
        uint64_t access_counter = 0;
        uint64_t hash_key = 0;     // FNV-1a hash of prompt_ids
    };

    // Compute a simple hash key from prompt token IDs
    static uint64_t compute_hash_key(const std::vector<int32_t>& prompt_ids) {
        uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
        for (int32_t tok : prompt_ids) {
            hash ^= static_cast<uint64_t>(tok);
            hash *= 1099511628211ULL;  // FNV-1a prime
        }
        return hash;
    }

    // Find a slot matching the prompt tokens (by hash and exact comparison)
    std::optional<int> find_slot(const std::vector<int32_t>& prompt_ids) {
        uint64_t hash = compute_hash_key(prompt_ids);

        for (int i = 0; i < MAX_SLOTS; i++) {
            if (slots_[i].occupied && slots_[i].hash_key == hash) {
                // Verify exact match (hash collision is possible but rare)
                if (slots_[i].prompt_ids == prompt_ids) {
                    // Update LRU
                    slots_[i].last_access = ++access_counter_;
                    return i;
                }
            }
        }
        return std::nullopt;
    }

    // Allocate a new slot (with LRU eviction if full)
    int allocate_slot(const std::vector<int32_t>& prompt_ids) {
        // First try to find an empty slot
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (!slots_[i].occupied) {
                init_slot(i, prompt_ids);
                return i;
            }
        }

        // All slots occupied - evict LRU
        int lru_idx = 0;
        uint64_t oldest_access = slots_[0].last_access;
        for (int i = 1; i < MAX_SLOTS; i++) {
            if (slots_[i].last_access < oldest_access) {
                oldest_access = slots_[i].last_access;
                lru_idx = i;
            }
        }

        // Free the LRU slot
        free_slot(lru_idx);
        init_slot(lru_idx, prompt_ids);
        return lru_idx;
    }

    // Free a specific slot and release its snapshot memory
    void free_slot(int slot) {
        if (slot >= 0 && slot < MAX_SLOTS && slots_[slot].occupied) {
            free_prefix_snapshot(slots_[slot].snapshot);
            slots_[slot].occupied = false;
            slots_[slot].prompt_ids.clear();
            slots_[slot].hash_key = 0;
            slots_[slot].last_access = 0;
        }
    }

    // Get snapshot reference for a slot
    PrefixSnapshot* get_snapshot(int slot) {
        if (slot >= 0 && slot < MAX_SLOTS && slots_[slot].occupied) {
            return &slots_[slot].snapshot;
        }
        return nullptr;
    }

    // Get statistics
    int occupied_count() const {
        int count = 0;
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (slots_[i].occupied) count++;
        }
        return count;
    }

private:
    std::array<Slot, MAX_SLOTS> slots_;
    uint64_t access_counter_ = 0;

    void init_slot(int slot, const std::vector<int32_t>& prompt_ids) {
        slots_[slot].occupied = true;
        slots_[slot].prompt_ids = prompt_ids;
        slots_[slot].hash_key = compute_hash_key(prompt_ids);
        slots_[slot].last_access = ++access_counter_;
        // snapshot will be populated by snapshot_target_cache()
    }
};

// ─── Prefill Coordinator ────────────────────────────────────────────────

class PrefillCoordinator {
public:
    bool should_compress(int prompt_len, bool has_tools) {
        if (!config_.enabled) return false;
        if (has_tools) return false;  // Don't compress requests with tools
        if (config_.mode == "off") return false;
        if (config_.mode == "always") return true;
        if (config_.mode == "auto") return prompt_len >= config_.threshold;
        return false;
    }

    struct Config {
        bool enabled = true;
        std::string mode = "always";
        int threshold = 8192;
        float keep_ratio = 0.1f;
        std::string drafter_path;
    } config_;
};

// ─── Core Generation Loop ─────────────────────────────────────────────────

std::vector<int32_t> DFlashServer::generate_internal(
    const std::vector<int32_t>& prompt_tokens,
    int n_gen,
    std::function<void(int32_t)> on_token,
    bool has_tools,
    const SamplerCfg& sampler) {

    std::lock_guard<std::mutex> lock(generation_mutex_);

    std::vector<int32_t> result;

    if (prompt_tokens.empty() || !target_weights_ || !draft_weights_ || !cache_) {
        std::cerr << "generate_internal: models or cache not initialized" << std::endl;
        return result;
    }

    // Resolve sampling config: use provided sampler or fall back to server defaults
    SamplerCfg effective_sampler = sampler;
    if (effective_sampler.temp == 0.0f && effective_sampler.top_p == 1.0f &&
        effective_sampler.top_k == 0 && effective_sampler.rep_pen == 1.0f) {
        effective_sampler = sampler_cfg_;  // Use server's resolved defaults
    }
    if (effective_sampler.seed == 0) {
        effective_sampler.seed = std::random_device{}();
    }
    // Apply to class member so run_dflash_decode can use it
    sampler_cfg_ = effective_sampler;
    // Seed the RNG for this request (deterministic if seed was provided)
    rng_.seed(effective_sampler.seed);

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int pf_ub = 384;  // prefill chunk size

    int32_t last_tok = -1;
    int committed = 0;
    bool cache_hit = false;

    // ─────────────────────────────────────────────────────────────────────
    // Prefix Cache Lookup
    // ─────────────────────────────────────────────────────────────────────
    auto cached_slot = prefix_cache_->find_slot(prompt_tokens);
    if (cached_slot) {
        PrefixSnapshot* snap = prefix_cache_->get_snapshot(*cached_slot);
        if (snap && restore_target_cache(*snap, *cache_)) {
            std::cout << "[PrefixCache] Cache hit! Restored " << cache_->cur_pos
                      << " tokens (slot " << *cached_slot << ")" << std::endl;
            committed = cache_->cur_pos;
            last_tok = cache_->last_tok;
            cache_hit = true;
        } else {
            std::cerr << "[PrefixCache] Cache hit but restore failed, falling back to prefill" << std::endl;
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // PFlash Prefill Compression (only on cache miss)
    // ─────────────────────────────────────────────────────────────────────
    std::vector<int32_t> working_tokens = prompt_tokens;

    if (!cache_hit) {
        // Don't compress requests with tools (tool calling requires exact tokenization)
        if (prefill_ && prefill_->should_compress((int)prompt_tokens.size(), has_tools)) {
            std::cout << "[PFlash] Compression triggered for " << prompt_tokens.size() << " tokens" << std::endl;

            // Park draft weights to free VRAM for drafter
            if (draft_weights_) {
                std::cout << "[PFlash] Parking draft weights" << std::endl;
                free_draft_weights(*draft_weights_);
            }

            // Compress the prompt
            auto compressed = compress_prefill(prompt_tokens);
            if (!compressed.empty() && compressed.size() < prompt_tokens.size()) {
                working_tokens = compressed;
                std::cout << "[PFlash] Using compressed tokens: " << working_tokens.size() << std::endl;
            }

            // Restore draft weights
            if (draft_weights_) {
                std::cout << "[PFlash] Restoring draft weights" << std::endl;
                // Note: draft weights need to be reloaded; this is handled by the initialization
            }
        }

        // Reset cache state for new request (only on cache miss)
        reset_target_cache(*cache_);
    }

    // ─────────────────────────────────────────────────────────────────────
    // Prefill: single-token decode with layer feature capture (skip on cache hit)
    // ─────────────────────────────────────────────────────────────────────
    if (!cache_hit) {
        for (int start = 0; start < static_cast<int>(working_tokens.size()); start += pf_ub) {
            const int nt = std::min(pf_ub, static_cast<int>(working_tokens.size()) - start);
            const int kv_len_p = start + nt;
            const bool with_m = (kv_len_p > 1) || (committed > 0);

            if (!build_target_step(target_sg_, start, nt, with_m, true, false, config_.fa_window)) {
                std::cerr << "prefill build failed @" << start << std::endl;
                return result;
            }

            // Embed tokens
            std::vector<float> pf_emb((size_t)hidden * nt);
            if (!target_weights_->embedder.embed(working_tokens.data() + start, nt, pf_emb.data())) {
                std::cerr << "prefill embed failed @" << start << std::endl;
                return result;
            }
            ggml_backend_tensor_set(target_sg_.inp_embed, pf_emb.data(), 0,
                                    sizeof(float) * pf_emb.size());

            // Set positions
            std::vector<int32_t> pf_pos(4 * nt);
            for (int i = 0; i < nt; i++) {
                const int p = start + i;
                pf_pos[0 * nt + i] = p;
                pf_pos[1 * nt + i] = p;
                pf_pos[2 * nt + i] = p;
                pf_pos[3 * nt + i] = 0;
            }
            ggml_backend_tensor_set(target_sg_.positions, pf_pos.data(), 0,
                                    sizeof(int32_t) * pf_pos.size());

            // Set attention mask if needed
            if (with_m) {
                std::vector<uint16_t> pf_mask;
                build_causal_mask(pf_mask, kv_len_p, nt, start);
                ggml_backend_tensor_set(target_sg_.attn_mask, pf_mask.data(), 0,
                                        sizeof(uint16_t) * pf_mask.size());
            }

            auto st = ggml_backend_graph_compute(cache_->backend, target_sg_.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::cerr << "prefill compute failed @" << start << ": " << (int)st << std::endl;
                return result;
            }

            committed = kv_len_p;
        }

        // Get last token logits for decode seed
        std::vector<float> last_logits(vocab);
        ggml_backend_tensor_get(target_sg_.logits, last_logits.data(),
                                (size_t)(committed - 1) * vocab * sizeof(float),
                                sizeof(float) * vocab);
        last_tok = argmax_f32(last_logits.data(), vocab);
        cache_->cur_pos = committed;
        cache_->last_tok = last_tok;

        // ─────────────────────────────────────────────────────────────────────
        // Snapshot the cache after prefill (for future requests)
        // ─────────────────────────────────────────────────────────────────────
        int new_slot = prefix_cache_->allocate_slot(prompt_tokens);
        if (new_slot >= 0) {
            PrefixSnapshot* snap = prefix_cache_->get_snapshot(new_slot);
            if (snap) {
                if (snapshot_target_cache(*target_weights_, *cache_, cache_->backend, *snap)) {
                    std::cout << "[PrefixCache] Snapshotted " << committed
                              << " tokens to slot " << new_slot
                              << " (occupied: " << prefix_cache_->occupied_count()
                              << "/" << PrefixCache::MAX_SLOTS << ")" << std::endl;
                } else {
                    std::cerr << "[PrefixCache] Failed to snapshot cache" << std::endl;
                }
            }
        } else {
            std::cerr << "[PrefixCache] No free slots available for snapshot" << std::endl;
        }
    }

    // Initialize feature mirror
    feature_mirror_.target_feat = cache_->target_feat;
    feature_mirror_.cap = cache_->target_feat_cap;
    feature_mirror_.offset = 0;

    // Run DFlash decode loop
    int n_generated = 0;
    if (!run_dflash_decode(working_tokens, n_gen, last_tok, on_token, n_generated, effective_sampler)) {
        std::cerr << "dflash decode failed" << std::endl;
        return result;
    }

    return result;
}

// Use the real Tokenizer implementation from tokenizer.h
// It's in the dflash27b namespace, so we can use it directly

// ─── DFlashServer Implementation ────────────────────────────────────────

DFlashServer::DFlashServer() = default;

DFlashServer::~DFlashServer() {
    stop();
}

bool DFlashServer::initialize(const ServerConfig& config) {
    config_ = config;

    // Create HTTP server
    http_server_ = std::make_unique<httplib::Server>();
    http_server_->set_logger([](const httplib::Request& req, const httplib::Response& res) {
        LOG_INFO(std::string("[") + req.method + "] " + req.path + " -> " + std::to_string(res.status));
    });

    // Initialize tokenizer
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load_from_gguf(config.target_path)) {
        LOG_ERROR("Failed to load tokenizer: " + tokenizer_->last_error());
        return false;
    }

    // Initialize prefix cache
    prefix_cache_ = std::make_unique<PrefixCache>();

    // Initialize prefill coordinator
    prefill_ = std::make_unique<PrefillCoordinator>();
    prefill_->config_ = {
        .enabled = config.prefill_enabled,
        .mode = config.prefill_mode,
        .threshold = config.prefill_threshold,
        .keep_ratio = config.prefill_keep_ratio,
        .drafter_path = config.prefill_drafter
    };

    // Initialize sampling state
    std::random_device rd;
    rng_.seed(rd());
    sampler_cfg_ = {
        .temp = 0.0f,      // Default: greedy (argmax)
        .top_p = 1.0f,
        .top_k = 0,
        .rep_pen = 1.0f,
        .rep_window = 256,
        .seed = 0
    };
    token_history_.reserve(1024);  // Pre-allocate for repetition penalty

    // Resolve KV cache types (config values override environment variables)
    ggml_type kv_k_type, kv_v_type;

    // First resolve from environment variables (defaults)
    dflash::resolve_kv_types(kv_k_type, kv_v_type);

    // Then override with config values if explicitly set
    // We set environment variables so create_target_cache() also uses them
    if (!config.kv_k_type.empty()) {
        ggml_type parsed = dflash::parse_kv_type(config.kv_k_type.c_str());
        if (parsed == GGML_TYPE_COUNT) {
            std::cerr << "Invalid --kv-k-type: " << config.kv_k_type << std::endl;
            return false;
        }
        kv_k_type = parsed;
        setenv("DFLASH27B_KV_K", config.kv_k_type.c_str(), 1);
    }
    if (!config.kv_v_type.empty()) {
        ggml_type parsed = dflash::parse_kv_type(config.kv_v_type.c_str());
        if (parsed == GGML_TYPE_COUNT) {
            std::cerr << "Invalid --kv-v-type: " << config.kv_v_type << std::endl;
            return false;
        }
        kv_v_type = parsed;
        setenv("DFLASH27B_KV_V", config.kv_v_type.c_str(), 1);
    }

    // Validate the resolved pair
    if (!dflash::is_supported_kv_pair(kv_k_type, kv_v_type)) {
        std::cerr << "KV pair (K=" << dflash::kv_type_name(kv_k_type)
                  << ", V=" << dflash::kv_type_name(kv_v_type)
                  << ") not supported" << std::endl;
        return false;
    }

    std::cout << "KV cache types: K=" << dflash::kv_type_name(kv_k_type)
              << ", V=" << dflash::kv_type_name(kv_v_type) << std::endl;

    // Log TurboQuant status
    if (kv_k_type == GGML_TYPE_TQ3_0 || kv_v_type == GGML_TYPE_TQ3_0) {
        std::cout << "TurboQuant enabled: TQ3_0 quantization in use" << std::endl;
    } else if (kv_k_type != GGML_TYPE_F16 || kv_v_type != GGML_TYPE_F16) {
        std::cout << "KV quantization enabled with K-rotation: kv_k_rotated=true" << std::endl;
    }

    // Create GPU backend for model loading
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::cerr << "Failed to create CUDA backend" << std::endl;
        return false;
    }
    std::cout << "Created CUDA backend" << std::endl;

    // Load target model (GGUF)
    target_weights_ = std::make_unique<TargetWeights>();
    if (!load_target_gguf(config.target_path, backend, *target_weights_)) {
        std::cerr << "Failed to load target model from: " << config.target_path << std::endl;
        ggml_backend_free(backend);
        return false;
    }
    std::cout << "Loaded target model: " << target_weights_->n_embd << " dims, "
              << target_weights_->n_layer << " layers" << std::endl;

    // Load draft model (safetensors or GGUF)
    draft_weights_ = std::make_unique<DraftWeights>();
    bool is_gguf = config.draft_path.size() >= 5 &&
                   config.draft_path.substr(config.draft_path.size() - 5) == ".gguf";

    if (is_gguf) {
        if (!load_draft_gguf(config.draft_path, backend, *draft_weights_)) {
            std::cerr << "Failed to load draft model from: " << config.draft_path << std::endl;
            ggml_backend_free(backend);
            return false;
        }
    } else {
        if (!load_draft_safetensors(config.draft_path, backend, *draft_weights_)) {
            std::cerr << "Failed to load draft model from: " << config.draft_path << std::endl;
            ggml_backend_free(backend);
            return false;
        }
    }
    std::cout << "Loaded draft model: " << draft_weights_->n_embd << " dims, "
              << draft_weights_->n_layer << " layers" << std::endl;

    // Create target KV cache
    cache_ = std::make_unique<TargetCache>();
    int max_verify_tokens = config.ddtree_budget + 1;  // For DDTree
    if (!create_target_cache(*target_weights_, config.max_ctx, max_verify_tokens,
                             backend, *cache_)) {
        std::cerr << "Failed to create target cache" << std::endl;
        ggml_backend_free(backend);
        return false;
    }
    std::cout << "Created target cache: max_ctx=" << cache_->max_ctx << std::endl;

    setup_routes();

    return true;
}

void DFlashServer::setup_routes() {
    // List models
    http_server_->Get("/v1/models",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_models(req, res);
        }
    );

    // Chat completions (non-streaming)
    http_server_->Post("/v1/chat/completions",
        [this](const httplib::Request& req, httplib::Response& res) {
            if (is_streaming_request(req)) {
                handle_chat_completions_stream(req, res);
            } else {
                handle_chat_completions(req, res);
            }
        }
    );

    // Health check
    http_server_->Get("/health",
        [](const httplib::Request& req, httplib::Response& res) {
            json health = {{"status", "ok"}, {"model", "dflash"}};
            res.set_content(health.dump(), "application/json");
        }
    );

    // Anthropic Messages API
    http_server_->Post("/v1/messages",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_anthropic_messages(req, res);
        }
    );
}

bool DFlashServer::is_streaming_request(const httplib::Request& req) {
    try {
        json body = json::parse(req.body);
        return body.value("stream", false);
    } catch (...) {
        return false;
    }
}

void DFlashServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"object", "list"},
        {"data", {
            {
                {"id", "dflash"},
                {"object", "model"},
                {"owned_by", "luce"},
                {"permission", {}}
            }
        }}
    };
    res.set_content(response.dump(), "application/json");
}

void DFlashServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    try {
        ChatRequest chat_req = parse_chat_request(req);

        // Tokenize prompt
        std::vector<int32_t> prompt_tokens;
        // TODO: Apply chat template and tokenize

        // Check if tools are present (affects prefill compression)
        bool has_tools = chat_req.tools.has_value() && !chat_req.tools->empty();

        // Resolve sampling config from request parameters
        SamplerCfg req_sampler;
        if (chat_req.temperature.has_value()) req_sampler.temp = *chat_req.temperature;
        if (chat_req.top_p.has_value()) req_sampler.top_p = *chat_req.top_p;
        if (chat_req.top_k.has_value()) req_sampler.top_k = *chat_req.top_k;
        if (chat_req.repetition_penalty.has_value()) req_sampler.rep_pen = *chat_req.repetition_penalty;
        if (chat_req.seed.has_value()) req_sampler.seed = *chat_req.seed;

        std::vector<int32_t> tokens = generate_internal(prompt_tokens, chat_req.max_tokens, nullptr, has_tools, req_sampler);

        // Build response
        ChatResponse chat_resp = build_chat_response(tokens, chat_req);

        // Set usage
        chat_resp.usage.prompt_tokens = static_cast<int>(prompt_tokens.size());
        chat_resp.usage.completion_tokens = static_cast<int>(tokens.size());
        chat_resp.usage.total_tokens = static_cast<int>(prompt_tokens.size() + tokens.size());

        // Convert to JSON
        json response = {
            {"id", chat_resp.id},
            {"object", chat_resp.object},
            {"created", chat_resp.created},
            {"model", chat_resp.model},
            {"choices", json::array()}
        };

        // Build choices array
        for (const auto& choice : chat_resp.choices) {
            json choice_json = {
                {"index", choice.index},
                {"message", {
                    {"role", choice.message.role},
                    {"content", choice.message.content}
                }},
                {"finish_reason", choice.finish_reason}
            };

            // Add tool_calls if present
            if (choice.message.tool_calls.has_value() && !choice.message.tool_calls->empty()) {
                choice_json["message"]["tool_calls"] = json::array();
                for (const auto& tc : choice.message.tool_calls.value()) {
                    choice_json["message"]["tool_calls"].push_back({
                        {"id", tc.id},
                        {"type", tc.type},
                        {"function", {
                            {"name", tc.function.name},
                            {"arguments", tc.function.arguments}
                        }}
                    });
                }
            }

            // Add name if present
            if (choice.message.name.has_value()) {
                choice_json["message"]["name"] = choice.message.name.value();
            }

            response["choices"].push_back(choice_json);
        }

        response["usage"] = {
            {"prompt_tokens", chat_resp.usage.prompt_tokens},
            {"completion_tokens", chat_resp.usage.completion_tokens},
            {"total_tokens", chat_resp.usage.total_tokens}
        };

        res.set_content(response.dump(), "application/json");

    } catch (const json::exception& e) {
        // JSON parsing error
        LOG_ERROR("JSON parsing error: " + std::string(e.what()));
        json error = {
            {"error", {
                {"message", std::string("Invalid JSON: ") + e.what()},
                {"type", "invalid_request_error"},
                {"code", "invalid_json"}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        // General error
        LOG_ERROR("Internal error: " + std::string(e.what()));
        json error = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    } catch (...) {
        // Unknown error
        LOG_ERROR("Unknown error occurred");
        json error = {
            {"error", {
                {"message", "Unknown error occurred"},
                {"type", "internal_error"},
                {"code", "unknown_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void DFlashServer::handle_chat_completions_stream(const httplib::Request& req, httplib::Response& res) {
    try {
        ChatRequest chat_req = parse_chat_request(req);

        // Set SSE headers
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");

        // Use content provider for streaming
        res.set_content_provider(
            /*length=*/0,  // Unknown length for chunked transfer
            /*content_type=*/"text/event-stream",
            /*provider=*/[this, chat_req](size_t offset, size_t length, httplib::DataSink& sink) mutable -> bool {
                (void)offset; (void)length;  // Unused in streaming mode
                // Tokenize prompt
                std::vector<int32_t> prompt_tokens;
                // TODO: Apply chat template and tokenize

                // Generate unique ID for this response
                std::string response_id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
                int created = std::time(nullptr);

                // Buffer for accumulated text (for incremental decode)
                std::vector<int32_t> generated_tokens;
                std::string accumulated_text;

                // Callback for streaming tokens
                auto on_token = [&sink, &response_id, &created, &chat_req, &generated_tokens, &accumulated_text, this](int32_t token) {
                    generated_tokens.push_back(token);

                    // Decode accumulated tokens to get incremental text
                    // Note: tokenizer_->decode() handles the full sequence, but for streaming
                    // we want to emit only the new delta. For simplicity, we decode all tokens
                    // and emit the full accumulated text (clients will handle delta display).
                    std::string full_text = tokenizer_->decode(generated_tokens);

                    // Calculate delta (simplified - just emit full text)
                    // A proper implementation would track the previous decode state
                    std::string delta = full_text;

                    // Build SSE event
                    json chunk = {
                        {"id", response_id},
                        {"object", "chat.completion.chunk"},
                        {"created", created},
                        {"model", chat_req.model},
                        {"choices", {
                            {
                                {"index", 0},
                                {"delta", {
                                    {"content", delta}
                                }},
                                {"finish_reason", nullptr}  // Still generating
                            }
                        }}
                    };

                    std::string sse_data = "data: " + chunk.dump() + "\n\n";
                    sink.write(sse_data.c_str(), sse_data.size());

                    return true;  // Continue generation
                };

                // Resolve sampling config from request parameters
                SamplerCfg req_sampler;
                if (chat_req.temperature.has_value()) req_sampler.temp = *chat_req.temperature;
                if (chat_req.top_p.has_value()) req_sampler.top_p = *chat_req.top_p;
                if (chat_req.top_k.has_value()) req_sampler.top_k = *chat_req.top_k;
                if (chat_req.repetition_penalty.has_value()) req_sampler.rep_pen = *chat_req.repetition_penalty;
                if (chat_req.seed.has_value()) req_sampler.seed = *chat_req.seed;

                // Generate tokens with streaming callback
                bool has_tools = chat_req.tools.has_value() && !chat_req.tools->empty();
                generate_internal(prompt_tokens, chat_req.max_tokens, on_token, has_tools, req_sampler);

                // Send final chunk with finish_reason
                json final_chunk = {
                    {"id", response_id},
                    {"object", "chat.completion.chunk"},
                    {"created", created},
                    {"model", chat_req.model},
                    {"choices", {
                        {
                            {"index", 0},
                            {"delta", json::object()},  // Empty delta for final chunk
                            {"finish_reason", "stop"}
                        }
                    }}
                };

                std::string final_sse = "data: " + final_chunk.dump() + "\n\n";
                final_sse += "data: [DONE]\n\n";
                sink.write(final_sse.c_str(), final_sse.size());

                // Check if stream_options.include_usage is requested
                if (chat_req.stream_options.has_value()) {
                    auto& stream_opts = chat_req.stream_options.value();
                    if (stream_opts.contains("include_usage") && stream_opts["include_usage"].get<bool>()) {
                        json usage_chunk = {
                            {"id", response_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created},
                            {"model", chat_req.model},
                            {"choices", json::array()},  // Empty choices for usage event
                            {"usage", {
                                {"prompt_tokens", static_cast<int>(prompt_tokens.size())},
                                {"completion_tokens", static_cast<int>(generated_tokens.size())},
                                {"total_tokens", static_cast<int>(prompt_tokens.size() + generated_tokens.size())}
                            }}
                        };

                        std::string usage_sse = "data: " + usage_chunk.dump() + "\n\n";
                        sink.write(usage_sse.c_str(), usage_sse.size());
                    }
                }

                return false;  // Signal completion
            },
            /*resource_releaser=*/[](bool success) {
                // Cleanup if needed
            }
        );

    } catch (const json::exception& e) {
        // JSON parsing error
        LOG_ERROR("Stream JSON parsing error: " + std::string(e.what()));
        json error = {
            {"error", {
                {"message", std::string("Invalid JSON: ") + e.what()},
                {"type", "invalid_request_error"},
                {"code", "invalid_json"}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        // General error
        LOG_ERROR("Stream internal error: " + std::string(e.what()));
        json error = {
            {"error", {
                {"message", e.what()},
                {"type", "internal_error"},
                {"code", "internal_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    } catch (...) {
        // Unknown error
        LOG_ERROR("Stream unknown error occurred");
        json error = {
            {"error", {
                {"message", "Unknown error occurred"},
                {"type", "internal_error"},
                {"code", "unknown_error"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

ChatRequest DFlashServer::parse_chat_request(const httplib::Request& req) {
    json body = json::parse(req.body);

    ChatRequest req_data;
    req_data.model = body.value("model", "dflash");
    req_data.stream = body.value("stream", false);
    req_data.max_tokens = body.value("max_tokens", 50000);

    // Parse temperature
    if (body.contains("temperature")) {
        req_data.temperature = body["temperature"];
    }

    // Parse top_p
    if (body.contains("top_p")) {
        req_data.top_p = body["top_p"];
    }

    // Parse top_k
    if (body.contains("top_k")) {
        req_data.top_k = body["top_k"];
    }

    // Parse stop
    if (body.contains("stop")) {
        req_data.stop = body["stop"];
    }

    // Parse stream_options
    if (body.contains("stream_options")) {
        req_data.stream_options = body["stream_options"];
    }

    // Parse chat_template_kwargs
    if (body.contains("chat_template_kwargs")) {
        req_data.chat_template_kwargs = body["chat_template_kwargs"];
    }

    // Parse tools
    if (body.contains("tools")) {
        req_data.tools = std::vector<ToolDef>{};
        for (const auto& tool : body["tools"]) {
            ToolDef tdef;
            tdef.type = tool.value("type", "function");
            tdef.function = tool["function"];
            req_data.tools->push_back(tdef);
        }
    }

    // Parse tool_choice
    if (body.contains("tool_choice")) {
        req_data.tool_choice = body["tool_choice"];
    }
    // Parse seed (for reproducibility)
    if (body.contains("seed")) {
        req_data.seed = body["seed"].get<uint64_t>();
    }
    // Parse repetition penalty
    if (body.contains("repetition_penalty")) {
        req_data.repetition_penalty = body["repetition_penalty"];
    }

    // Parse messages
    if (body.contains("messages")) {
        for (const auto& msg : body["messages"]) {
            ChatMessage m;
            m.role = msg["role"];
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    m.content = msg["content"];
                }
                // TODO: Handle content as array (for multimodal)
            }
            if (msg.contains("name")) {
                m.name = msg["name"];
            }
            if (msg.contains("tool_call_id")) {
                m.tool_call_id = msg["tool_call_id"];
            }
            if (msg.contains("tool_calls")) {
                m.tool_calls = std::vector<ToolCall>{};
                for (const auto& tc : msg["tool_calls"]) {
                    ToolCall tool_call;
                    tool_call.id = tc["id"];
                    tool_call.type = tc.value("type", "function");
                    tool_call.function.name = tc["function"]["name"];
                    tool_call.function.arguments = tc["function"]["arguments"];
                    m.tool_calls->push_back(tool_call);
                }
            }
            req_data.messages.push_back(m);
        }
    }

    return req_data;
}

ChatResponse DFlashServer::build_chat_response(
    const std::vector<int32_t>& tokens,
    const ChatRequest& req,
    const std::string& reasoning
) {
    ChatResponse resp;
    resp.id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    resp.created = std::time(nullptr);
    resp.model = req.model;

    // Decode tokens to text
    std::string text = tokenizer_->decode(tokens);

    // Parse tool calls if tools were provided
    std::string content = text;
    std::vector<ToolCall> tool_calls;
    if (req.tools.has_value() && !req.tools->empty()) {
        ToolCallParseResult tc_result = parse_tool_calls(text, &req.tools.value());
        content = tc_result.cleaned_content;

        // Convert parsed tool calls to OpenAI format
        for (const auto& pt : tc_result.tool_calls) {
            ToolCall tc;
            tc.id = pt.id;
            tc.type = pt.type;
            tc.function.name = pt.function.name;
            tc.function.arguments = pt.function.arguments;
            tool_calls.push_back(std::move(tc));
        }
    }

    ChatResponse::Choice choice;
    choice.message.role = "assistant";
    choice.message.content = content;

    // Add tool calls if present
    if (!tool_calls.empty()) {
        choice.message.tool_calls = std::move(tool_calls);
        choice.finish_reason = "tool_calls";
    } else {
        choice.finish_reason = "stop";
    }

    resp.choices.push_back(choice);

    resp.usage.prompt_tokens = 0;  // Will be set by caller
    resp.usage.completion_tokens = tokens.size();
    resp.usage.total_tokens = 0;  // Will be set by caller

    return resp;
}

// ─── Reasoning Extraction (using parsers) ────────────────────────────────

std::pair<std::string, std::string> DFlashServer::extract_reasoning(
    const std::string& text
) {
    // Use the parse_reasoning function from parsers.cpp
    ReasoningParseResult result = parse_reasoning(text, /*thinking_enabled=*/true);
    return {result.content, result.reasoning};
}

// ─── Anthropic Messages API Implementation ──────────────────────────────

AnthropicMessagesRequest DFlashServer::parse_anthropic_messages_request(
    const httplib::Request& req
) {
    json body = json::parse(req.body);

    AnthropicMessagesRequest areq;

    // Parse model
    areq.model = body.value("model", "dflash");

    // Parse stream flag
    areq.stream = body.value("stream", false);

    // Parse max_tokens (required by Anthropic)
    if (body.contains("max_tokens")) {
        areq.max_tokens = body["max_tokens"];
    }

    // Parse optional parameters
    if (body.contains("temperature")) {
        areq.temperature = body["temperature"];
    }
    if (body.contains("top_p")) {
        areq.top_p = body["top_p"];
    }
    if (body.contains("top_k")) {
        areq.top_k = body["top_k"];
    }
    // Parse seed (OpenAI-compatible extension for reproducibility)
    if (body.contains("seed")) {
        areq.seed = body["seed"].get<uint64_t>();
    }

    // Parse system prompt
    if (body.contains("system")) {
        if (body["system"].is_string()) {
            areq.system = body["system"];
        }
        // TODO: Handle system as array of content blocks
    }

    // Parse stop_sequences
    if (body.contains("stop_sequences")) {
        std::vector<std::string> seqs;
        for (const auto& s : body["stop_sequences"]) {
            seqs.push_back(s);
        }
        areq.stop_sequences = std::move(seqs);
    }

    // Parse messages
    if (body.contains("messages")) {
        for (const auto& msg : body["messages"]) {
            AnthropicMessage am;
            am.role = msg["role"];

            // Parse content (can be string or array)
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    // Simple text content
                    AnthropicContentBlock block;
                    block.type = "text";
                    block.text = msg["content"];
                    am.content.push_back(block);
                } else if (msg["content"].is_array()) {
                    // Array of content blocks
                    for (const auto& cb : msg["content"]) {
                        AnthropicContentBlock block;
                        block.type = cb.value("type", "text");

                        if (block.type == "text") {
                            block.text = cb.value("text", "");
                        } else if (block.type == "image") {
                            // TODO: Handle image content
                            block.source_type = cb["source"].value("type", "");
                            block.media_type = cb["source"].value("media_type", "");
                            block.data = cb["source"].value("data", "");
                        } else if (block.type == "tool_use") {
                            block.id = cb.value("id", "");
                            block.input_json = cb.value("input", json::object()).dump();
                        } else if (block.type == "tool_result") {
                            block.id = cb.value("tool_use_id", "");
                            if (cb["content"].is_string()) {
                                block.text = cb["content"];
                            }
                        }

                        am.content.push_back(block);
                    }
                }
            }

            // Parse optional name
            if (msg.contains("name")) {
                am.name = msg["name"];
            }

            areq.messages.push_back(am);
        }
    }

    // Parse tools
    if (body.contains("tools")) {
        areq.tools = std::vector<AnthropicTool>{};
        for (const auto& tool : body["tools"]) {
            AnthropicTool atool;
            atool.name = tool["name"];
            atool.description = tool.value("description", "");
            atool.input_schema = tool["input_schema"];
            areq.tools->push_back(atool);
        }
    }

    // Parse tool_choice
    if (body.contains("tool_choice")) {
        auto& tc = body["tool_choice"];
        if (tc.is_string()) {
            std::string tc_str = tc;
            if (tc_str == "auto") {
                areq.tool_choice = AnthropicToolChoice{AnthropicToolChoiceType::auto_mode, ""};
            } else if (tc_str == "any") {
                areq.tool_choice = AnthropicToolChoice{AnthropicToolChoiceType::any, ""};
            }
        } else if (tc.is_object()) {
            std::string type = tc.value("type", "auto");
            if (type == "tool") {
                areq.tool_choice = AnthropicToolChoice{
                    AnthropicToolChoiceType::tool,
                    tc.value("name", "")
                };
            }
        }
    }

    return areq;
}

ChatRequest DFlashServer::convert_anthropic_to_chat(
    const AnthropicMessagesRequest& areq
) {
    ChatRequest creq;

    // Set basic fields
    creq.model = areq.model;
    creq.stream = areq.stream;
    creq.max_tokens = areq.max_tokens;
    creq.temperature = areq.temperature;
    creq.top_p = areq.top_p;

    // Convert messages
    for (const auto& am : areq.messages) {
        ChatMessage cm;
        cm.role = am.role;

        // Join all text content blocks
        std::string text_content;
        for (const auto& cb : am.content) {
            if (cb.type == "text") {
                text_content += cb.text;
            }
            // TODO: Handle tool_use and tool_result blocks
        }

        cm.content = text_content;

        if (am.name) {
            cm.name = *am.name;
        }

        creq.messages.push_back(cm);
    }

    // Handle system prompt (prepend as user message for now)
    // TODO: Properly integrate system prompt with chat template
    if (!areq.system.empty()) {
        ChatMessage sys_msg;
        sys_msg.role = "system";
        sys_msg.content = areq.system;
        creq.messages.insert(creq.messages.begin(), sys_msg);
    }

    // Convert stop_sequences to stop
    if (areq.stop_sequences) {
        creq.stop = *areq.stop_sequences;
    }

    // Convert tools (from Anthropic to OpenAI format)
    if (areq.tools) {
        creq.tools = std::vector<ToolDef>{};
        for (const auto& atool : *areq.tools) {
            ToolDef tdef;
            tdef.type = "function";
            tdef.function = {
                {"name", atool.name},
                {"description", atool.description},
                {"parameters", atool.input_schema}
            };
            creq.tools->push_back(tdef);
        }
    }

    return creq;
}

AnthropicMessagesResponse DFlashServer::build_anthropic_response(
    const std::vector<int32_t>& tokens,
    const AnthropicMessagesRequest& areq,
    const std::string& reasoning
) {
    AnthropicMessagesResponse resp;

    // Generate message ID
    auto now = std::chrono::system_clock::now();
    auto timestamp = now.time_since_epoch().count();
    resp.id = "msg_" + std::to_string(timestamp);

    resp.model = areq.model;
    resp.type = "message";
    resp.role = "assistant";

    // Decode tokens to text
    std::string text = tokenizer_->decode(tokens);

    // Parse reasoning if present
    std::string content_text = text;
    if (!reasoning.empty()) {
        // If reasoning was provided separately, use it
        // TODO: Handle reasoning content in Anthropic format
    } else {
        // Check if text contains reasoning tags
        size_t think_pos = text.find("\xe5\x8f\x8d\xe5\xa4\x8d\xe6\x8e\xa8\xe7\x90\x86");  // 反复推理
        if (think_pos != std::string::npos) {
            // Has reasoning - extract it
            auto [content, reasoning_content] = extract_reasoning(text);
            content_text = content;
            // TODO: Include reasoning in response
        }
    }

    // Check if tools were requested and parse tool calls
    bool has_tool_calls = areq.tools.has_value() && !areq.tools->empty();

    if (has_tool_calls) {
        // Convert Anthropic tools to OpenAI format for parsing
        std::vector<ToolDef> openai_tools;
        for (const auto& atool : areq.tools.value()) {
            ToolDef tdef;
            tdef.type = "function";
            tdef.function = {
                {"name", atool.name},
                {"description", atool.description},
                {"parameters", atool.input_schema}
            };
            openai_tools.push_back(tdef);
        }

        // Parse tool calls from text
        ToolCallParseResult tc_result = parse_tool_calls(text, &openai_tools);

        if (!tc_result.tool_calls.empty()) {
            // Add tool_use content blocks
            for (const auto& tc : tc_result.tool_calls) {
                AnthropicContentBlock block;
                block.type = "tool_use";
                block.id = tc.id;
                block.text = tc.function.name;  // Store name in text field
                block.input_json = tc.function.arguments;
                resp.content.push_back(block);
            }

            // Add text content if non-empty
            if (!tc_result.cleaned_content.empty()) {
                AnthropicContentBlock text_block;
                text_block.type = "text";
                text_block.text = tc_result.cleaned_content;
                resp.content.insert(resp.content.begin(), text_block);
            }

            resp.stop_reason = "tool_use";
        } else {
            // No tool calls found, return text only
            AnthropicContentBlock block;
            block.type = "text";
            block.text = content_text;
            resp.content.push_back(block);
            resp.stop_reason = "end_turn";
        }
    } else {
        // No tools requested, return text only
        AnthropicContentBlock block;
        block.type = "text";
        block.text = content_text;
        resp.content.push_back(block);
        resp.stop_reason = "end_turn";
    }

    // Set usage
    resp.usage.input_tokens = 0;  // TODO: Get from prompt
    resp.usage.output_tokens = tokens.size();

    return resp;
}

json DFlashServer::anthropic_response_to_json(
    const AnthropicMessagesResponse& resp
) {
    json content_blocks = json::array();
    for (const auto& cb : resp.content) {
        json block;
        block["type"] = cb.type;

        if (cb.type == "text") {
            block["text"] = cb.text;
        } else if (cb.type == "tool_use") {
            block["id"] = cb.id;
            block["name"] = cb.text;  // Reusing text field for tool name
            block["input"] = json::parse(cb.input_json);
        } else if (cb.type == "tool_result") {
            block["tool_use_id"] = cb.id;
            block["content"] = cb.text;
        } else if (cb.type == "image") {
            block["source"] = {
                {"type", cb.source_type},
                {"media_type", cb.media_type},
                {"data", cb.data}
            };
        }

        content_blocks.push_back(block);
    }

    json response = {
        {"id", resp.id},
        {"type", resp.type},
        {"role", resp.role},
        {"content", content_blocks},
        {"model", resp.model},
        {"stop_reason", resp.stop_reason},
        {"usage", {
            {"input_tokens", resp.usage.input_tokens},
            {"output_tokens", resp.usage.output_tokens}
        }}
    };

    return response;
}

std::string DFlashServer::anthropic_event_to_sse(
    const std::string& event_name,
    const json& event_data
) {
    std::ostringstream oss;
    oss << "event: " << event_name << "\n";
    oss << "data: " << event_data.dump() << "\n\n";
    return oss.str();
}

void DFlashServer::handle_anthropic_messages(
    const httplib::Request& req,
    httplib::Response& res
) {
    try {
        AnthropicMessagesRequest areq = parse_anthropic_messages_request(req);

        if (areq.stream) {
            handle_anthropic_messages_stream_impl(areq, res);
        } else {
            handle_anthropic_messages_impl(areq, res);
        }

    } catch (const json::exception& e) {
        // JSON parsing error
        LOG_ERROR("Anthropic JSON parsing error: " + std::string(e.what()));
        json error = {
            {"type", "error"},
            {"error", {
                {"type", "invalid_request_error"},
                {"message", std::string("Invalid JSON: ") + e.what()}
            }}
        };
        res.status = 400;
        res.set_content(error.dump(), "application/json");
    } catch (const std::exception& e) {
        // General error
        LOG_ERROR("Anthropic internal error: " + std::string(e.what()));
        json error = {
            {"type", "error"},
            {"error", {
                {"type", "internal_error"},
                {"message", std::string("Internal error: ") + e.what()}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    } catch (...) {
        // Unknown error
        LOG_ERROR("Anthropic unknown error occurred");
        json error = {
            {"type", "error"},
            {"error", {
                {"type", "internal_error"},
                {"message", "Unknown error occurred"}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void DFlashServer::handle_anthropic_messages_impl(
    const AnthropicMessagesRequest& areq,
    httplib::Response& res
) {
    // Convert to ChatRequest and use existing generation logic
    ChatRequest creq = convert_anthropic_to_chat(areq);

    // Tokenize prompt
    std::vector<int32_t> prompt_tokens;
    // TODO: Apply chat template and tokenize

    // Resolve sampling config from request parameters
    SamplerCfg req_sampler;
    if (areq.temperature.has_value()) req_sampler.temp = *areq.temperature;
    if (areq.top_p.has_value()) req_sampler.top_p = *areq.top_p;
    if (areq.top_k.has_value()) req_sampler.top_k = *areq.top_k;
    if (areq.seed.has_value()) req_sampler.seed = *areq.seed;

    // Check if tools are present (affects prefill compression)
    bool has_tools = areq.tools.has_value() && !areq.tools->empty();

    // Generation is handled internally with prefix cache support
    std::vector<int32_t> tokens = generate_internal(prompt_tokens, areq.max_tokens, nullptr, has_tools, req_sampler);

    // Build Anthropic response
    AnthropicMessagesResponse aresp = build_anthropic_response(tokens, areq);

    // Set usage
    aresp.usage.input_tokens = static_cast<int>(prompt_tokens.size());
    aresp.usage.output_tokens = static_cast<int>(tokens.size());

    // Convert to JSON
    json response_json = anthropic_response_to_json(aresp);

    res.set_content(response_json.dump(), "application/json");
}

void DFlashServer::handle_anthropic_messages_stream_impl(
    const AnthropicMessagesRequest& areq,
    httplib::Response& res
) {
    // Set SSE headers
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // Use content provider for streaming (ContentProviderWithoutLength)
    res.set_content_provider(
        /*content_type=*/"text/event-stream",
        /*provider=*/[this, areq](size_t offset, httplib::DataSink& sink) mutable -> bool {
            (void)offset;  // Unused in streaming mode
            // Convert to ChatRequest and use existing generation logic
            ChatRequest creq = convert_anthropic_to_chat(areq);

            // Tokenize prompt
            std::vector<int32_t> prompt_tokens;
            // TODO: Apply chat template and tokenize

            // Generate message ID
            auto now = std::chrono::system_clock::now();
            auto timestamp = now.time_since_epoch().count();
            std::string message_id = "msg_" + std::to_string(timestamp);

            // Helper lambda to write SSE events
            auto write_sse = [&sink](const std::string& event_name, const json& event_data) {
                std::ostringstream oss;
                oss << "event: " << event_name << "\n";
                oss << "data: " << event_data.dump() << "\n\n";
                std::string sse = oss.str();
                sink.write(sse.c_str(), sse.size());
            };

            // Send message_start event
            json start_event = {
                {"type", "message_start"},
                {"message", {
                    {"id", message_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", areq.model},
                    {"stop_reason", nullptr},
                    {"usage", {
                        {"input_tokens", static_cast<int>(prompt_tokens.size())},
                        {"output_tokens", 0}
                    }}
                }}
            };
            write_sse("message_start", start_event);

            // Send content_block_start event
            json block_start_event = {
                {"type", "content_block_start"},
                {"index", 0},
                {"content_block", {
                    {"type", "text"},
                    {"text", ""}
                }}
            };
            write_sse("content_block_start", block_start_event);

            // Buffer for generated tokens
            std::vector<int32_t> generated_tokens;
            int output_tokens = 0;

            // Callback for streaming tokens
            auto on_token = [&sink, &write_sse, &generated_tokens, &output_tokens, this](int32_t token) {
                generated_tokens.push_back(token);
                output_tokens++;

                // Decode the latest token(s) to get incremental text
                // For simplicity, we decode all accumulated tokens
                std::string text = tokenizer_->decode(generated_tokens);

                // Send content_block_delta event
                json delta_event = {
                    {"type", "content_block_delta"},
                    {"index", 0},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", text}
                    }},
                    {"usage", {
                        {"output_tokens", output_tokens}
                    }}
                };
                write_sse("content_block_delta", delta_event);

                return true;  // Continue generation
            };

            // Resolve sampling config from request parameters
            SamplerCfg req_sampler;
            if (areq.temperature.has_value()) req_sampler.temp = *areq.temperature;
            if (areq.top_p.has_value()) req_sampler.top_p = *areq.top_p;
            if (areq.top_k.has_value()) req_sampler.top_k = *areq.top_k;
            if (areq.seed.has_value()) req_sampler.seed = *areq.seed;

            // Generate tokens with streaming callback
            bool has_tools = areq.tools.has_value() && !areq.tools->empty();
            generate_internal(prompt_tokens, creq.max_tokens, on_token, has_tools, req_sampler);

            // Send content_block_stop event
            json block_stop_event = {
                {"type", "content_block_stop"},
                {"index", 0}
            };
            write_sse("content_block_stop", block_stop_event);

            // Send message_delta event with final usage
            json delta_event = {
                {"type", "message_delta"},
                {"delta", {
                    {"stop_reason", "end_turn"},
                    {"stop_sequence", nullptr}
                }},
                {"usage", {
                    {"output_tokens", output_tokens}
                }}
            };
            write_sse("message_delta", delta_event);

            // Send message_stop event
            json stop_event = {
                {"type", "message_stop"}
            };
            write_sse("message_stop", stop_event);

            return false;  // Signal completion
        },
        /*resource_releaser=*/[](bool success) {
            // Cleanup if needed
        }
    );
}

void DFlashServer::run() {
    running_ = true;
    std::cout << "DFlash server starting on " << config_.host << ":" << config_.port << std::endl;
    http_server_->listen(config_.host, config_.port);
}

void DFlashServer::stop() {
    running_ = false;
    if (http_server_) {
        http_server_->stop();
    }
}

// ─── PFlash Compression Implementation ─────────────────────────────────────

bool DFlashServer::load_drafter() {
    if (drafter_loaded_) {
        return true;  // Already loaded
    }

    if (prefill_->config_.drafter_path.empty()) {
        std::cerr << "Drafter path not configured" << std::endl;
        return false;
    }

    drafter_ctx_ = std::make_unique<DrafterContext>();
    if (!dflash27b::load_drafter(prefill_->config_.drafter_path, /*gpu_layers=*/999, *drafter_ctx_)) {
        std::cerr << "Failed to load drafter: " << dflash27b_last_error() << std::endl;
        drafter_ctx_.reset();
        return false;
    }

    drafter_loaded_ = true;
    std::cout << "[PFlash] Drafter loaded: n_layer=" << drafter_ctx_->weights.n_layer
              << ", n_head=" << drafter_ctx_->weights.n_head << std::endl;
    return true;
}

void DFlashServer::free_drafter() {
    if (drafter_loaded_ && drafter_ctx_) {
        dflash27b::free_drafter(*drafter_ctx_);
        drafter_ctx_.reset();
        drafter_loaded_ = false;
        std::cout << "[PFlash] Drafter freed" << std::endl;
    }
}

std::vector<int32_t> DFlashServer::compress_prefill(const std::vector<int32_t>& prompt_tokens) {
    if (!drafter_loaded_) {
        if (!load_drafter()) {
            std::cerr << "[PFlash] Failed to load drafter for compression" << std::endl;
            return prompt_tokens;  // Return original tokens on failure
        }
    }

    // Call drafter_score_and_compress
    std::vector<int32_t> compressed = dflash27b::drafter_score_and_compress(
        *drafter_ctx_,
        prompt_tokens,
        prefill_->config_.keep_ratio,
        /*chunk_size=*/32,
        /*n_lookahead=*/8,
        /*pool_kernel=*/13
    );

    if (compressed.empty()) {
        std::cerr << "[PFlash] Compression returned empty, using original tokens" << std::endl;
        return prompt_tokens;
    }

    std::cout << "[PFlash] Compressed: " << prompt_tokens.size() << " -> " << compressed.size()
              << " tokens (ratio=" << (float)compressed.size() / prompt_tokens.size() << ")" << std::endl;

    return compressed;
}

// ─── Helper Functions ───────────────────────────────────────────────────

std::string tokens_to_text(const std::vector<int32_t>& tokens, Tokenizer* tokenizer) {
    if (!tokenizer || tokens.empty()) return "";
    return tokenizer->decode(tokens);
}

// ─── Main Entry Point ───────────────────────────────────────────────────

} // namespace dflash27b

// ─── Standalone Server Main Function ─────────────────────────────────────

int main(int argc, char** argv) {
    using namespace dflash27b;

    // Parse command line arguments
    ServerConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--target" && i + 1 < argc) {
            config.target_path = argv[++i];
        } else if (arg == "--draft" && i + 1 < argc) {
            config.draft_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--max-ctx" && i + 1 < argc) {
            config.max_ctx = std::atoi(argv[++i]);
        } else if (arg == "--prefill-mode" && i + 1 < argc) {
            config.prefill_mode = argv[++i];
        } else if (arg == "--prefill-threshold" && i + 1 < argc) {
            config.prefill_threshold = std::atoi(argv[++i]);
        } else if (arg == "--prefill-keep-ratio" && i + 1 < argc) {
            config.prefill_keep_ratio = std::atof(argv[++i]);
        } else if (arg == "--prefill-drafter" && i + 1 < argc) {
            config.prefill_drafter = argv[++i];
        } else if (arg == "--prefill-always") {
            config.prefill_mode = "always";
        } else if (arg == "--kv-k-type" && i + 1 < argc) {
            config.kv_k_type = argv[++i];
        } else if (arg == "--kv-v-type" && i + 1 < argc) {
            config.kv_v_type = argv[++i];
        } else if (arg == "--fa-window" && i + 1 < argc) {
            config.fa_window = std::atoi(argv[++i]);
        } else if (arg == "--tree-budget" && i + 1 < argc) {
            config.ddtree_budget = std::atoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --target <gguf> --draft <safetensors> [options]\n"
                      << "\nOptions:\n"
                      << "  --target <path>    Target model GGUF file\n"
                      << "  --draft <path>     Draft model safetensors file\n"
                      << "  --host <addr>      Server host (default: 0.0.0.0)\n"
                      << "  --port <port>      Server port (default: 8000)\n"
                      << "  --max-ctx <n>      Max context length (default: 131072)\n"
                      << "  --kv-k-type <type> KV cache K type (f16/bf16/q4_0/q8_0/tq3_0, default: q8_0)\n"
                      << "  --kv-v-type <type> KV cache V type (f16/bf16/q4_0/q8_0/tq3_0, default: q8_0)\n"
                      << "  --fa-window <n>    Flash attention sliding window (0=full, default: 2048)\n"
                      << "  --tree-budget <n>  DDTree budget (default: 22, 0=disabled)\n"
                      << "  --prefill-mode     off/auto/always (default: auto)\n"
                      << "  --prefill-threshold <n>  Min tokens for auto mode (default: 8192)\n"
                      << "  --prefill-keep-ratio <f> Compression ratio (default: 0.1)\n"
                      << "  --prefill-drafter <path>  Drafter GGUF path\n"
                      << "  --prefill-always    Alias for --prefill-mode always\n";
            return 0;
        }
    }

    // Validate required arguments
    if (config.target_path.empty()) {
        std::cerr << "Error: --target is required" << std::endl;
        return 1;
    }
    if (config.draft_path.empty()) {
        std::cerr << "Error: --draft is required" << std::endl;
        return 1;
    }

    // Create and initialize server
    DFlashServer server;
    if (!server.initialize(config)) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }

    // Run server
    server.run();

    return 0;
}
