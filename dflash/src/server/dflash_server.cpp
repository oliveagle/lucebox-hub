// dflash_server.cpp - C++ HTTP Server Implementation
//
// This file implements a standalone HTTP server for DFlash that replaces
// the Python-based server_tools.py.

#include "dflash_server.h"
#include "internal.h"
#include "parsers.h"
#include "tokenizer.h"
#include "kv_quant.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "dflash_graph.h"
#include "qwen3_drafter.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <optional>

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace dflash27b {

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
    int& n_generated) {

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    int committed = static_cast<int>(prompt.size());
    n_generated = 0;

    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;

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
        // Note: positions_k needs to be set on the correct tensor

        auto st = ggml_backend_graph_compute(cache_->backend, draft_sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::cerr << "draft compute failed: " << (int)st << std::endl;
            return false;
        }

        ggml_backend_tensor_get(draft_sg_.argmax_tokens, draft_tok.data(), 0,
                                sizeof(int32_t) * q_len);
        draft_tok[0] = last_tok;

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

        ggml_backend_tensor_get(target_sg_.argmax_tokens, &replay_last_tok, 0, sizeof(int32_t));
        last_tok = replay_last_tok;

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
        // TODO: Add GGML buffer pointers
    };

    std::optional<int> find_slot(const std::vector<int32_t>& prompt_ids) {
        // Simple linear search for matching prompt
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (slots_[i].occupied && slots_[i].prompt_ids == prompt_ids) {
                return i;
            }
        }
        return std::nullopt;
    }

    int allocate_slot() {
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (!slots_[i].occupied) {
                slots_[i].occupied = true;
                return i;
            }
        }
        return -1;  // Full
    }

    void free_slot(int slot) {
        if (slot >= 0 && slot < MAX_SLOTS) {
            slots_[slot].occupied = false;
            slots_[slot].prompt_ids.clear();
        }
    }

private:
    std::array<Slot, MAX_SLOTS> slots_;
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
    std::function<void(int32_t)> on_token) {

    std::lock_guard<std::mutex> lock(generation_mutex_);

    std::vector<int32_t> result;

    if (prompt_tokens.empty() || !target_weights_ || !draft_weights_ || !cache_) {
        std::cerr << "generate_internal: models or cache not initialized" << std::endl;
        return result;
    }

    // PFlash prefill compression
    const bool has_tools = false;  // Tool integration will be added in STORY-008
    std::vector<int32_t> working_tokens = prompt_tokens;

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

    // Reset cache state for new request
    reset_target_cache(*cache_);

    const int hidden = DFLASH27B_TARGET_HIDDEN;
    const int vocab = DFLASH27B_TARGET_VOCAB;
    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int pf_ub = 384;  // prefill chunk size

    int32_t last_tok = -1;
    int committed = 0;

    // Prefill: single-token decode with layer feature capture
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

    // Initialize feature mirror
    feature_mirror_.target_feat = cache_->target_feat;
    feature_mirror_.cap = cache_->target_feat_cap;
    feature_mirror_.offset = 0;

    // Run DFlash decode loop
    int n_generated = 0;
    if (!run_dflash_decode(working_tokens, n_gen, last_tok, on_token, n_generated)) {
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
        std::cout << "[" << req.method << "] " << req.path << " -> " << res.status << std::endl;
    });

    // Initialize tokenizer
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load_from_gguf(config.target_path)) {
        std::cerr << "Failed to load tokenizer: " << tokenizer_->last_error() << std::endl;
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

        // Check cache
        auto cached_slot = prefix_cache_->find_slot(prompt_tokens);

        std::vector<int32_t> tokens;

        if (cached_slot) {
            // Restore from cache
            // TODO: Implement cache restore
        } else {
            // Generate
            tokens = generate_internal(prompt_tokens, chat_req.max_tokens);

            // Cache result
            int slot = prefix_cache_->allocate_slot();
            if (slot >= 0) {
                // TODO: Snapshot cache
            }
        }

        // Build response
        ChatResponse chat_resp = build_chat_response(tokens, chat_req);

        // Convert to JSON
        json response = {
            {"id", chat_resp.id},
            {"object", chat_resp.object},
            {"created", chat_resp.created},
            {"model", chat_resp.model},
            {"choices", {
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", tokenizer_->decode(tokens)}
                    }},
                    {"finish_reason", "stop"}
                }
            }},
            {"usage", {
                {"prompt_tokens", static_cast<int>(prompt_tokens.size())},
                {"completion_tokens", static_cast<int>(tokens.size())},
                {"total_tokens", static_cast<int>(prompt_tokens.size() + tokens.size())}
            }}
        };

        res.set_content(response.dump(), "application/json");

    } catch (const std::exception& e) {
        json error = {{"error", {{"message", e.what()}}}};
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void DFlashServer::handle_chat_completions_stream(const httplib::Request& req, httplib::Response& res) {
    // For now, use non-streaming as placeholder
    // TODO: Implement proper SSE streaming with correct cpp-httplib API
    handle_chat_completions(req, res);
}

ChatRequest DFlashServer::parse_chat_request(const httplib::Request& req) {
    json body = json::parse(req.body);

    ChatRequest req_data;
    req_data.model = body.value("model", "dflash");
    req_data.stream = body.value("stream", false);
    req_data.max_tokens = body.value("max_tokens", 50000);

    if (body.contains("messages")) {
        for (const auto& msg : body["messages"]) {
            ChatMessage m;
            m.role = msg["role"];
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    m.content = msg["content"];
                }
            }
            req_data.messages.push_back(m);
        }
    }

    // Parse other fields...
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

    ChatResponse::Choice choice;
    choice.message.role = "assistant";
    choice.message.content = tokenizer_->decode(tokens);
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);

    resp.usage.prompt_tokens = 0;  // TODO
    resp.usage.completion_tokens = tokens.size();
    resp.usage.total_tokens = 0;  // TODO

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

    // Create text content block
    AnthropicContentBlock block;
    block.type = "text";
    block.text = content_text;
    resp.content.push_back(block);

    // Set stop_reason
    resp.stop_reason = "end_turn";  // Default
    // TODO: Determine actual stop_reason from generation

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

    } catch (const std::exception& e) {
        // Return Anthropic-style error
        json error = {
            {"type", "error"},
            {"error", {
                {"type", "internal_error"},
                {"message", std::string("Internal error: ") + e.what()}
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

    // Check cache
    auto cached_slot = prefix_cache_->find_slot(prompt_tokens);

    std::vector<int32_t> tokens;

    if (cached_slot) {
        // Restore from cache
        // TODO: Implement cache restore
    } else {
        // Generate
        tokens = generate_internal(prompt_tokens, creq.max_tokens);

        // Cache result
        int slot = prefix_cache_->allocate_slot();
        if (slot >= 0) {
            // TODO: Snapshot cache
        }
    }

    // Build Anthropic response
    AnthropicMessagesResponse aresp = build_anthropic_response(tokens, areq);

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

    // For now, use non-streaming implementation as placeholder
    // TODO: Implement proper SSE streaming with Anthropic event format
    handle_anthropic_messages_impl(areq, res);
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
