#include "qwen35_backend.h"
#include "qwen35_dflash_target.h"
#include "graph_builders.h"
#include "feature_copy.h"
#include "peer_access.h"
#include "attn_masks.h"
#include "common/sampler.h"
#include "common/io_utils.h"
#include "qwen3/qwen3_drafter.h"
#include "triattention_runner.h"

#include "ggml-cuda.h"

// ggml-cuda dequantize: Q8_0/F16/BF16 → F32
using to_fp32_cuda_t = void (*)(const void *, float *, int64_t, cudaStream_t);
to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sstream>

namespace dflash27b {

using dflash27b::argmax_f32;

#define IS_EOS_TOK(tok, w)                                         \
    ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id)                  \
   || ((w).eos_id      >= 0 && (tok) == (w).eos_id     ) )

// ── Construction / destruction ──────────────────────────────────────────

Qwen35Backend::Qwen35Backend(const Qwen35Config & cfg) : cfg_(cfg) {}

Qwen35Backend::~Qwen35Backend() { shutdown(); }

// ── init() ──────────────────────────────────────────────────────────────

bool Qwen35Backend::init() {
    split_gpus_ = (cfg_.device.gpu != cfg_.draft_gpu);

    target_backend_ = ggml_backend_cuda_init(cfg_.device.gpu);
    if (!target_backend_) {
        std::fprintf(stderr, "target cuda init failed\n");
        return false;
    }
    draft_backend_ = target_backend_;
    if (split_gpus_) {
        draft_backend_ = ggml_backend_cuda_init(cfg_.draft_gpu);
        if (!draft_backend_) {
            std::fprintf(stderr, "draft cuda init failed\n");
            return false;
        }
    }
    if (split_gpus_ && g_peer_access_opt_in) {
        enable_peer_access_pair(cfg_.device.gpu, cfg_.draft_gpu);
    }

    // Load target
    if (!load_target_gguf(cfg_.target_path, target_backend_, w_)) {
        std::fprintf(stderr, "target load: %s\n", dflash27b_last_error());
        return false;
    }
    std::printf("[target] %s\n", dflash27b_last_error());

    // Load draft
    if (cfg_.draft_path) {
        std::string dp(cfg_.draft_path);
        bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
            ? load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, &w_)
            : load_draft_safetensors(cfg_.draft_path, draft_backend_, dw_, &w_);
        if (!draft_ok) {
            std::fprintf(stderr, "draft load: %s\n", dflash27b_last_error());
            return false;
        }
        std::printf("[draft]  loaded\n");

        if (cfg_.draft_swa_window > 0) {
            dw_.swa_window = cfg_.draft_swa_window;
            for (int il = 0; il < dw_.n_layer - 1; il++)
                dw_.layers[il].is_swa = true;
            std::printf("[draft]  SWA layers: %d/%d (window=%d)\n",
                        dw_.n_layer - 1, dw_.n_layer, dw_.swa_window);
        }
    }

    // Create KV cache
    const int max_verify_tokens = cfg_.ddtree_mode
        ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
        : dw_.block_size;
    if (!create_target_cache(w_, cfg_.device.max_ctx, max_verify_tokens, target_backend_, cache_,
                             /*prefill_only=*/true)) {
        std::fprintf(stderr, "cache: %s\n", dflash27b_last_error());
        return false;
    }

    // Optionally init feature mirror
    if (cfg_.use_feature_mirror && split_gpus_) {
        const int mirror_cap = std::min(cfg_.draft_ctx_max, cfg_.device.max_ctx);
        if (!draft_feature_mirror_init(feature_mirror_, draft_backend_,
                                       cfg_.draft_gpu, cfg_.device.gpu, mirror_cap)) {
            std::fprintf(stderr, "warning: feature mirror init failed, using direct copies\n");
        }
    }

    return true;
}

// ── print_ready_banner ──────────────────────────────────────────────────

void Qwen35Backend::print_ready_banner() const {
    std::printf("[daemon] ready\n");
    std::fflush(stdout);
}

// ── Park / unpark ───────────────────────────────────────────────────────

bool Qwen35Backend::park(const std::string & what) {
    bool want_draft  = (what.empty() || what == "all" || what == "draft");
    bool want_target = (what.empty() || what == "all" || what == "target");

    if (want_draft && !draft_parked_) {
        step_graph_destroy(draft_sg_);
        free_draft_weights(dw_);
        draft_parked_ = true;
        std::printf("[park] draft released\n"); std::fflush(stdout);
    }
    if (want_target && !target_parked_) {
        step_graph_destroy(proj_sg_);
        free_target_weights(w_);
        target_parked_ = true;
        std::printf("[park] target released\n"); std::fflush(stdout);
    }
    return true;
}

bool Qwen35Backend::unpark(const std::string & what) {
    bool want_target = (what.empty() || what == "all" || what == "target");
    bool want_draft  = (what.empty() || what == "all" || what == "draft");

    if (want_target && target_parked_) {
        if (!load_target_gguf(cfg_.target_path, target_backend_, w_)) {
            std::fprintf(stderr, "[unpark] target: %s\n", dflash27b_last_error());
            return false;
        }
        target_parked_ = false;
        std::printf("[unpark] target restored\n"); std::fflush(stdout);
    }
    if (want_draft && draft_parked_ && cfg_.draft_path) {
        std::string dp(cfg_.draft_path);
        bool draft_ok = (dp.size() >= 5 && dp.substr(dp.size() - 5) == ".gguf")
            ? load_draft_gguf(cfg_.draft_path, draft_backend_, dw_, &w_)
            : load_draft_safetensors(cfg_.draft_path, draft_backend_, dw_, &w_);
        if (!draft_ok) {
            std::fprintf(stderr, "[unpark] draft: %s\n", dflash27b_last_error());
            return false;
        }
        if (cfg_.draft_swa_window > 0) {
            dw_.swa_window = cfg_.draft_swa_window;
            for (int il = 0; il < dw_.n_layer - 1; il++)
                dw_.layers[il].is_swa = true;
        }
        draft_parked_ = false;
        std::printf("[unpark] draft restored\n"); std::fflush(stdout);
    }
    return true;
}

// ── Snapshots ───────────────────────────────────────────────────────────

bool Qwen35Backend::snapshot_save(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    snapshot_free(slot);
    PrefixSnapshot & snap = prefix_snapshots_[slot];
    return snapshot_target_cache(w_, cache_, target_backend_, snap);
}

void Qwen35Backend::snapshot_free(int slot) {
    if (slot < 0 || slot >= PREFIX_SLOTS) return;
    free_prefix_snapshot(prefix_snapshots_[slot]);
}

bool Qwen35Backend::snapshot_used(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return false;
    return prefix_snapshots_[slot].ctx != nullptr;
}

int Qwen35Backend::snapshot_cur_pos(int slot) const {
    if (slot < 0 || slot >= PREFIX_SLOTS) return 0;
    return prefix_snapshots_[slot].cur_pos;
}

// ── Compress (pflash) ───────────────────────────────────────────────────

bool Qwen35Backend::handle_compress(const std::string & line, const DaemonIO & io) {
    // Lazy-load drafter on first use
    if (!drafter_loaded_) {
        std::fprintf(stderr, "[compress] loading drafter...\n");
        if (!load_drafter("/opt/lucebox/models/drafter/Qwen3-0.6B-BF16.gguf",
                          /*gpu_layers=*/999, drafter_ctx_)) {
            std::fprintf(stderr, "[compress] drafter init failed: %s\n",
                         dflash27b_last_error());
            io.emit(-1);
            return false;
        }
        drafter_loaded_ = true;
        std::fprintf(stderr, "[compress] drafter ready\n");
    }

    // Park target+draft to free VRAM for the drafter
    const bool was_target_parked = target_parked_;
    const bool was_draft_parked  = draft_parked_;
    if (!target_parked_) park("target");
    if (!draft_parked_)  park("draft");

    // Parse: "compress <n_draft> <prompt_path>"
    std::istringstream iss(line);
    std::string cmd;
    int n_draft = 0;
    std::string prompt_path;
    iss >> cmd >> n_draft >> prompt_path;

    bool ok = false;
    if (n_draft > 0 && !prompt_path.empty()) {
        std::vector<int32_t> tokens = read_int32_file(prompt_path);
        if (!tokens.empty()) {
            const float keep = (float)n_draft / (float)tokens.size();
            auto compressed = drafter_score_and_compress(drafter_ctx_, tokens, keep);
            ok = !compressed.empty();
            if (ok) {
                for (int32_t t : compressed) io.emit(t);
            }
        }
    }
    io.emit(-1);

    // Restore park state
    if (!was_target_parked) unpark("target");
    if (!was_draft_parked)  unpark("draft");

    return ok;
}

void Qwen35Backend::free_drafter() {
    if (drafter_loaded_) {
        dflash27b::free_drafter(drafter_ctx_);
        drafter_loaded_ = false;
        std::printf("[drafter] freed\n"); std::fflush(stdout);
    }
}

// ── try_handle_command (arch-specific) ──────────────────────────────────

bool Qwen35Backend::try_handle_command(const std::string & line, const DaemonIO & io) {
    // SNAPSHOT_THIN <slot> — lightweight snapshot (SSM state only, no KV copy)
    if (line.compare(0, 14, "SNAPSHOT_THIN ") == 0) {
        int slot = std::atoi(line.c_str() + 14);
        if (slot >= 0 && slot < PREFIX_SLOTS) {
            snapshot_free(slot);
            PrefixSnapshot & snap = prefix_snapshots_[slot];
            snapshot_target_cache_thin(w_, cache_, target_backend_,
                                       /*kv_start=*/0, /*kv_end=*/cache_.cur_pos, snap);
            std::printf("[snapshot_thin] slot=%d pos=%d\n", slot, snap.cur_pos);
            std::fflush(stdout);
        }
        io.emit(-1);
        return true;
    }

    return false;
}

// ── DFlash spec decode target ────────────────────────────────────────────

DFlashTarget * Qwen35Backend::dflash_target() {
    if (!dflash_target_) {
        dflash_target_ = std::make_unique<Qwen35DFlashTarget>(
            w_, cache_, target_backend_, sg_,
            cfg_.kq_stride_pad, cfg_.fa_window);
    }
    return dflash_target_.get();
}

// ── Shutdown ────────────────────────────────────────────────────────────

void Qwen35Backend::shutdown() {
    free_drafter();
    step_graph_destroy(sg_);
    step_graph_destroy(draft_sg_);
    step_graph_destroy(proj_sg_);
    draft_feature_mirror_free(feature_mirror_);
    for (int i = 0; i < PREFIX_SLOTS; i++) {
        free_prefix_snapshot(prefix_snapshots_[i]);
    }
    if (!target_parked_) free_target_weights(w_);
    if (!draft_parked_)  free_draft_weights(dw_);
    free_target_cache(cache_);
    if (split_gpus_ && draft_backend_) {
        ggml_backend_free(draft_backend_);
        draft_backend_ = nullptr;
    }
    if (target_backend_) {
        ggml_backend_free(target_backend_);
        target_backend_ = nullptr;
    }
}

// ── Generate (speculative decode) ───────────────────────────────────────

GenerateResult Qwen35Backend::generate(const GenerateRequest & req,
                                        const DaemonIO & io) {
    GenerateResult result;
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    // Prefill
    const int committed = do_prefill(req.prompt, io, req.snap_pos, req.snap_slot);
    if (committed < 0) {
        result.error = "prefill";
        return result;
    }

    // Decode (speculative)
    if (req.n_gen > 0) {
        if (!do_spec_decode(committed, req.n_gen, result.tokens, io)) {
            result.error = "decode";
            return result;
        }
    }

    result.ok = true;
    return result;
}

// ── Restore + generate ──────────────────────────────────────────────────

GenerateResult Qwen35Backend::restore_and_generate(int slot,
                                                    const GenerateRequest & req,
                                                    const DaemonIO & io) {
    GenerateResult result;
    if (slot < 0 || slot >= PREFIX_SLOTS || !prefix_snapshots_[slot].ctx) {
        result.error = "bad slot";
        io.emit(-1);
        return result;
    }

    // Restore snapshot
    restore_target_cache(prefix_snapshots_[slot], cache_);

    // Now generate from restored state
    sampler_ = req.sampler;
    if (req.do_sample && sampler_.seed != 0) {
        sampler_rng_.seed(sampler_.seed);
    }

    const int snap_pos = prefix_snapshots_[slot].cur_pos;

    // If there are additional prompt tokens beyond the snapshot, prefill them
    int committed = snap_pos;
    if (!req.prompt.empty()) {
        // The prompt here is the diff (tokens beyond the snapshot)
        committed = do_prefill(req.prompt, io, req.snap_pos, req.snap_slot);
        if (committed < 0) {
            result.error = "prefill";
            return result;
        }
    }

    // Decode
    if (req.n_gen > 0) {
        if (!do_spec_decode(committed, req.n_gen, result.tokens, io)) {
            result.error = "decode";
            return result;
        }
    }

    result.ok = true;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS — will be fleshed out when the spec-decode loop is
// migrated from test_dflash.cpp. For now, these are stubs that produce an
// error so the build succeeds and the interface is validated.
// ═══════════════════════════════════════════════════════════════════════════

int Qwen35Backend::do_prefill(const std::vector<int32_t> & tokens,
                               const DaemonIO & io,
                               int snap_pos, int snap_slot) {
    (void)io; (void)snap_pos; (void)snap_slot;

    const int hidden = w_.n_embd;
    const int PREFILL_UBATCH = 512;
    const int prompt_len = (int)tokens.size();

    // Migrate to full-mode KV cache if needed
    migrate_prefill_cache(w_, cfg_.device.max_ctx,
                          cfg_.ddtree_mode
                              ? std::max<int>(dw_.block_size, cfg_.ddtree_budget + 1)
                              : dw_.block_size,
                          target_backend_, cache_);

    // Chunked prefill
    std::vector<float> embed_buf((size_t)hidden * PREFILL_UBATCH);
    int committed = 0;
    for (int start = 0; start < prompt_len; start += PREFILL_UBATCH) {
        const int n_tokens = std::min(PREFILL_UBATCH, prompt_len - start);
        const bool with_mask = (cfg_.kq_stride_pad > KQ_MASK_PAD) || (n_tokens > 1);

        if (!build_target_step(sg_, w_, cache_, target_backend_,
                               /*kv_start=*/start, /*n_tokens=*/n_tokens,
                               with_mask, /*capture=*/true,
                               /*capture_delta_intermediate=*/false,
                               cfg_.fa_window,
                               /*last_token_logits_only=*/(start + n_tokens < prompt_len),
                               cfg_.kq_stride_pad)) {
            std::fprintf(stderr, "prefill build @%d\n", start);
            return -1;
        }

        // Embed
        if (!w_.embedder.embed(tokens.data() + start, n_tokens, embed_buf.data())) {
            return -1;
        }
        ggml_backend_tensor_set(sg_.inp_embed, embed_buf.data(), 0,
                                sizeof(float) * (size_t)hidden * n_tokens);

        // Positions (M-RoPE)
        std::vector<int32_t> pos_buf((size_t)4 * n_tokens, 0);
        for (int i = 0; i < n_tokens; i++) {
            const int p = start + i;
            pos_buf[4 * i + 0] = p;
            pos_buf[4 * i + 1] = p;
            pos_buf[4 * i + 2] = p;
            pos_buf[4 * i + 3] = 0;
        }
        ggml_backend_tensor_set(sg_.positions, pos_buf.data(), 0,
                                sizeof(int32_t) * pos_buf.size());

        // Mask
        if (sg_.attn_mask) {
            const int win_start = (cfg_.fa_window > 0 && start > cfg_.fa_window)
                                      ? (start - cfg_.fa_window) : 0;
            const int kv_len = start + n_tokens - win_start;
            std::vector<uint16_t> mask_buf;
            build_causal_mask(mask_buf, kv_len, n_tokens, start, cfg_.kq_stride_pad, win_start);
            ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0,
                                    sizeof(uint16_t) * mask_buf.size());
        }

        // Compute
        auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "prefill compute @%d failed\n", start);
            return -1;
        }

        // Snapshot at boundary if requested
        if (snap_pos >= 0 && snap_slot >= 0 &&
            start + n_tokens >= snap_pos && start < snap_pos) {
            // Not at boundary yet — next chunk will cross it
        } else if (snap_pos >= 0 && snap_slot >= 0 &&
                   start + n_tokens == snap_pos) {
            cache_.cur_pos = snap_pos;
            snapshot_save(snap_slot);
        }

        committed = start + n_tokens;
        cache_.cur_pos = committed;

        // Sync feature mirror if active
        if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_, feature_mirror_, start, n_tokens);
        }
    }

    return committed;
}

bool Qwen35Backend::do_spec_decode(int committed, int n_gen,
                                    std::vector<int32_t> & out_tokens,
                                    const DaemonIO & io) {
    // Full DFlash speculative decoding loop with TriAttention compression.
    // Based on test_dflash.cpp implementation adapted for Qwen35Backend.

    const int q_len = DFLASH27B_DRAFT_BLOCK_SIZE;
    const int hidden = w_.n_embd;
    const int vocab = w_.n_vocab;
    const int mask_tok = DFLASH27B_DRAFT_MASK_TOKEN_ID;

    // If no draft model, fall back to simple AR decode
    if (draft_parked_ || !cfg_.draft_path) {
        std::vector<float> logits_buf(vocab);
        std::vector<float> embed_buf(hidden);
        int32_t last_tok = out_tokens.empty() ? 0 : out_tokens.back();

        // Read argmax from prefill if available
        if (out_tokens.empty()) {
            int32_t argmax = 0;
            ggml_backend_tensor_get(sg_.argmax_tokens, &argmax, 0, sizeof(int32_t));
            last_tok = argmax;
            out_tokens.push_back(last_tok);
            io.emit(last_tok);
            if (IS_EOS_TOK(last_tok, w_)) { io.emit(-1); return true; }
            committed++;
        }

        for (int i = (out_tokens.size() > 1 ? 1 : 0); i < n_gen; i++) {
            if (!build_target_step(sg_, w_, cache_, target_backend_,
                                   /*kv_start=*/committed, /*n_tokens=*/1,
                                   /*with_mask=*/false, /*capture=*/true,
                                   /*capture_delta_intermediate=*/false,
                                   /*fa_window=*/0,
                                   /*last_token_logits_only=*/false,
                                   cfg_.kq_stride_pad)) {
                return false;
            }

            if (!w_.embedder.embed(&last_tok, 1, embed_buf.data())) return false;
            ggml_backend_tensor_set(sg_.inp_embed, embed_buf.data(), 0, sizeof(float) * hidden);
            int32_t pos4[4] = {committed, committed, committed, 0};
            ggml_backend_tensor_set(sg_.positions, pos4, 0, sizeof(int32_t) * 4);

            auto st = ggml_backend_graph_compute(target_backend_, sg_.gf);
            if (st != GGML_STATUS_SUCCESS) return false;

            ggml_backend_tensor_get(sg_.logits, logits_buf.data(), 0, sizeof(float) * vocab);
            int32_t next_tok;
            if (sampler_.temp > 0) {
                next_tok = sample_logits(logits_buf.data(), vocab, sampler_,
                                         out_tokens, sampler_rng_);
            } else {
                next_tok = 0;
                float best = logits_buf[0];
                for (int j = 1; j < vocab; j++) {
                    if (logits_buf[j] > best) { best = logits_buf[j]; next_tok = j; }
                }
            }

            out_tokens.push_back(next_tok);
            io.emit(next_tok);
            committed++;
            cache_.cur_pos = committed;
            last_tok = next_tok;

            if (IS_EOS_TOK(next_tok, w_)) break;
        }

        io.emit(-1);
        return true;
    }

    // === DFlash speculative decode loop ===
    int n_generated = 0;
    int n_draft_steps = 0;
    int n_accept_sum = 0;

    std::vector<float> noise_embed((size_t)hidden * q_len);
    std::vector<int32_t> noise_ids(q_len);
    std::vector<int32_t> draft_tok(q_len);
    std::vector<int32_t> target_tok(q_len);
    std::vector<int32_t> pos_q(q_len);
    std::vector<int32_t> pos_k;
    std::vector<uint16_t> mask_buf;
    std::vector<int32_t> pos4_buf(4 * q_len);

    // Get last_tok from out_tokens or prefill
    int32_t last_tok = out_tokens.empty() ? 0 : out_tokens.back();

    while (n_generated < n_gen) {
        const int need_commit_budget = n_gen - n_generated;

        // 1) Build noise block [last_tok, MASK*15]
        noise_ids[0] = last_tok;
        for (int i = 1; i < q_len; i++) noise_ids[i] = mask_tok;
        if (!w_.embedder.embed(noise_ids.data(), q_len, noise_embed.data())) {
            std::fprintf(stderr, "do_spec_decode: noise embed failed\n");
            return false;
        }

        // Draft context window
        constexpr int DRAFT_CTX_MAX = 2048;
        const int draft_ctx = std::min(committed, std::max(DRAFT_CTX_MAX, cfg_.draft_ctx_max));
        const int draft_start = committed - draft_ctx;
        int mirror_slot0 = 0;
        const bool use_mirror_view =
            cfg_.use_feature_mirror && split_gpus_ &&
            draft_feature_mirror_can_view(feature_mirror_, committed, draft_ctx, mirror_slot0);

        // 2) Draft forward
        if (!build_draft_step(draft_sg_, dw_, use_mirror_view ? nullptr : &w_,
                              draft_backend_, /*ctx_len=*/draft_ctx,
                              use_mirror_view ? &feature_mirror_ : nullptr,
                              committed)) {
            std::fprintf(stderr, "do_spec_decode: draft build failed\n");
            return false;
        }

        ggml_backend_tensor_set(draft_sg_.inp_embed, noise_embed.data(), 0,
                                sizeof(float) * noise_embed.size());

        if (!use_mirror_view && cfg_.use_feature_mirror) {
            if (!copy_feature_ring_range_to_tensor(feature_mirror_, draft_sg_.target_hidden_cat,
                                                   draft_start, draft_ctx)) {
                std::fprintf(stderr, "do_spec_decode: draft feature copy failed\n");
                return false;
            }
        }

        pos_k.resize((size_t)draft_ctx + q_len);
        for (int i = 0; i < q_len; i++) pos_q[i] = draft_ctx + i;
        for (int i = 0; i < draft_ctx + q_len; i++) pos_k[i] = i;
        ggml_backend_tensor_set(draft_sg_.positions, pos_q.data(), 0, sizeof(int32_t) * q_len);
        ggml_backend_tensor_set(draft_sg_.positions_k, pos_k.data(), 0, sizeof(int32_t) * pos_k.size());

        auto st = ggml_backend_graph_compute(draft_backend_, draft_sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "do_spec_decode: draft compute failed\n");
            return false;
        }

        // Get draft tokens via argmax
        if (split_gpus_) {
            // Cross-GPU: need projection step
            if (!proj_sg_.gf || !proj_sg_.hidden_input || proj_sg_.hidden_input->ne[1] != q_len) {
                if (!build_lm_head_projection_step(proj_sg_, w_, target_backend_, q_len)) {
                    std::fprintf(stderr, "do_spec_decode: projection build failed\n");
                    return false;
                }
            }
            const size_t hidden_bytes = ggml_nbytes(draft_sg_.hidden_states);
            if (!copy_peer_async(proj_sg_.hidden_input->data, cfg_.device.gpu,
                                 draft_sg_.hidden_states->data, cfg_.draft_gpu,
                                 hidden_bytes)) {
                std::fprintf(stderr, "do_spec_decode: hidden peer copy failed\n");
                return false;
            }
            cudaSetDevice(cfg_.device.gpu);
            cudaDeviceSynchronize();
            st = ggml_backend_graph_compute(target_backend_, proj_sg_.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "do_spec_decode: projection compute failed\n");
                return false;
            }
            std::vector<float> draft_logits_buf((size_t)vocab * q_len);
            ggml_backend_tensor_get(proj_sg_.logits, draft_logits_buf.data(), 0,
                                    sizeof(float) * vocab * q_len);
            for (int i = 0; i < q_len; i++) {
                draft_tok[i] = argmax_f32(draft_logits_buf.data() + (size_t)i * vocab, vocab);
            }
        } else {
            // Same GPU: direct argmax
            std::vector<int32_t> gpu_argmax(q_len);
            ggml_backend_tensor_get(draft_sg_.argmax_tokens, gpu_argmax.data(), 0,
                                    sizeof(int32_t) * q_len);
            for (int i = 0; i < q_len; i++) draft_tok[i] = gpu_argmax[i];
        }
        draft_tok[0] = last_tok;

        // 3) Snapshot SSM state for rollback
        if (!cfg_.fast_rollback) {
            snapshot_ssm_state(cache_);
        }

        // 4) Target verify (batch mode)
        const int verify_fa_window = cfg_.fa_window;
        if (!build_target_step(sg_, w_, cache_, target_backend_,
                                /*kv_start=*/committed, /*n_tokens=*/q_len,
                                /*with_mask=*/true, /*capture=*/true,
                                /*capture_delta_intermediate=*/cfg_.fast_rollback,
                                verify_fa_window,
                                /*last_token_logits_only=*/false,
                                cfg_.kq_stride_pad)) {
            std::fprintf(stderr, "do_spec_decode: verify build failed\n");
            return false;
        }

        std::vector<float> verify_embed(hidden * q_len);
        if (!w_.embedder.embed(draft_tok.data(), q_len, verify_embed.data())) {
            std::fprintf(stderr, "do_spec_decode: verify embed failed\n");
            return false;
        }
        ggml_backend_tensor_set(sg_.inp_embed, verify_embed.data(), 0,
                                sizeof(float) * verify_embed.size());

        for (int i = 0; i < q_len; i++) {
            int p = committed + i;
            pos4_buf[0 * q_len + i] = p;
            pos4_buf[1 * q_len + i] = p;
            pos4_buf[2 * q_len + i] = p;
            pos4_buf[3 * q_len + i] = 0;
        }
        ggml_backend_tensor_set(sg_.positions, pos4_buf.data(), 0, sizeof(int32_t) * 4 * q_len);

        {
            const int win_start_v = (verify_fa_window > 0 && committed > verify_fa_window)
                                        ? (committed - verify_fa_window) : 0;
            const int win_len_v = committed + q_len - win_start_v;
            build_causal_mask(mask_buf, win_len_v, q_len, committed, cfg_.kq_stride_pad, win_start_v);
        }
        ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());

        st = ggml_backend_graph_compute(target_backend_, sg_.gf);
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "do_spec_decode: verify compute failed\n");
            return false;
        }

        ggml_backend_tensor_get(sg_.argmax_tokens, target_tok.data(), 0,
                                sizeof(int32_t) * q_len);

        // 5) Greedy longest-prefix accept
        int accept_n = 1;
        for (int i = 0; i < q_len - 1; i++) {
            if (draft_tok[i + 1] == target_tok[i]) accept_n++;
            else break;
        }

        int bonus_tok = -1;
        int commit_n;
        if (cfg_.fast_rollback) {
            commit_n = accept_n;
        } else {
            if (accept_n < q_len) {
                bonus_tok = target_tok[accept_n - 1];
            }
            commit_n = accept_n + (bonus_tok >= 0 ? 1 : 0);
        }

        // Don't overshoot n_gen
        if (commit_n > need_commit_budget) {
            commit_n = need_commit_budget;
            if (commit_n <= accept_n) bonus_tok = -1;
        }

        // 6) Rollback and commit
        if (cfg_.fast_rollback) {
            // Fast rollback: use captured intermediates
            if (commit_n < q_len) {
                const int rollback_idx = commit_n - 1;
                const int n_delta = (int)sg_.delta_captures.size();
                cudaStream_t stream = nullptr;
                for (int il = 0; il < n_delta; il++) {
                    const DeltaNetCapture & cap = sg_.delta_captures[il];
                    if (!cap.ssm_intermediate_states || !cap.conv_input) {
                        std::fprintf(stderr, "do_spec_decode: missing capture at layer %d\n", il);
                        return false;
                    }

                    // SSM rollback
                    const size_t ssm_elems =
                        (size_t)cache_.ssm_state[il]->ne[0] *
                        (size_t)cache_.ssm_state[il]->ne[1] *
                        (size_t)cache_.ssm_state[il]->ne[2];
                    const size_t ssm_src_offset =
                        (size_t)rollback_idx * cap.ssm_intermediate_states->nb[3];
                    const void * ssm_src =
                        (const char *)cap.ssm_intermediate_states->data + ssm_src_offset;
                    ggml_get_to_fp32_cuda(cap.ssm_intermediate_states->type)(
                        ssm_src, (float *)cache_.ssm_state[il]->data,
                        (int64_t)ssm_elems, stream);

                    // Conv rollback
                    const int K_conv = 4;
                    const int row_cnt = (int)cap.conv_input->ne[1];
                    const size_t elt = ggml_element_size(cap.conv_input);
                    const size_t dpitch = (K_conv - 1) * elt;
                    const size_t spitch = cap.conv_input->nb[1];
                    const void * conv_src = (const char *)cap.conv_input->data + commit_n * elt;
                    cudaError_t ce = cudaMemcpy2DAsync(cache_.conv_state[il]->data, dpitch,
                                                       conv_src, spitch,
                                                       (K_conv - 1) * elt, row_cnt,
                                                       cudaMemcpyDeviceToDevice, stream);
                    if (ce != cudaSuccess) {
                        std::fprintf(stderr, "do_spec_decode: conv rollback il=%d: %s\n",
                                     il, cudaGetErrorString(ce));
                        return false;
                    }
                }
                cudaStreamSynchronize(stream);
            }
            last_tok = target_tok[commit_n - 1];

            // Emit accepted tokens
            for (int i = 0; i < commit_n; i++) {
                out_tokens.push_back(draft_tok[i]);
                io.emit(draft_tok[i]);
                if (IS_EOS_TOK(draft_tok[i], w_)) {
                    io.emit(-1);
                    return true;
                }
            }
        } else {
            // Legacy replay path
            restore_ssm_state(cache_);

            std::vector<int32_t> replay_tok(commit_n);
            for (int i = 0; i < commit_n; i++) {
                replay_tok[i] = (i < accept_n) ? draft_tok[i] : bonus_tok;
            }

            bool replay_with_mask = (commit_n > 1);
            const int replay_fa_window = cfg_.fa_window;
            if (!build_target_step(sg_, w_, cache_, target_backend_,
                                    committed, commit_n,
                                    replay_with_mask, /*capture=*/true,
                                    false, replay_fa_window,
                                    /*last_token_logits_only=*/false,
                                    cfg_.kq_stride_pad)) {
                std::fprintf(stderr, "do_spec_decode: replay build failed\n");
                return false;
            }

            std::vector<float> replay_embed(hidden * commit_n);
            if (!w_.embedder.embed(replay_tok.data(), commit_n, replay_embed.data())) {
                std::fprintf(stderr, "do_spec_decode: replay embed failed\n");
                return false;
            }
            ggml_backend_tensor_set(sg_.inp_embed, replay_embed.data(), 0, sizeof(float) * replay_embed.size());

            std::vector<int32_t> replay_pos(4 * commit_n);
            for (int i = 0; i < commit_n; i++) {
                int p = committed + i;
                replay_pos[0 * commit_n + i] = p;
                replay_pos[1 * commit_n + i] = p;
                replay_pos[2 * commit_n + i] = p;
                replay_pos[3 * commit_n + i] = 0;
            }
            ggml_backend_tensor_set(sg_.positions, replay_pos.data(), 0, sizeof(int32_t) * 4 * commit_n);

            if (replay_with_mask) {
                const int win_start_r = (replay_fa_window > 0 && committed > replay_fa_window)
                                            ? (committed - replay_fa_window) : 0;
                const int win_len_r = committed + commit_n - win_start_r;
                build_causal_mask(mask_buf, win_len_r, commit_n, committed, cfg_.kq_stride_pad, win_start_r);
                ggml_backend_tensor_set(sg_.attn_mask, mask_buf.data(), 0, sizeof(uint16_t) * mask_buf.size());
            }

            st = ggml_backend_graph_compute(target_backend_, sg_.gf);
            if (st != GGML_STATUS_SUCCESS) {
                std::fprintf(stderr, "do_spec_decode: replay compute failed\n");
                return false;
            }

            std::vector<float> last_logits(vocab);
            ggml_backend_tensor_get(sg_.logits, last_logits.data(),
                                    sizeof(float) * vocab * (commit_n - 1),
                                    sizeof(float) * vocab);
            last_tok = argmax_f32(last_logits.data(), vocab);

            // Emit replayed tokens
            for (int i = 0; i < commit_n; i++) {
                out_tokens.push_back(replay_tok[i]);
                io.emit(replay_tok[i]);
                if (IS_EOS_TOK(replay_tok[i], w_)) {
                    io.emit(-1);
                    return true;
                }
            }
        }

        // Sync feature mirror if active
        if (feature_mirror_.target_feat && !draft_parked_) {
            draft_feature_mirror_sync_range(cache_, feature_mirror_, committed, commit_n);
        }

        committed += commit_n;
        n_generated += commit_n;
        n_accept_sum += accept_n;
        cache_.cur_pos = committed;
        n_draft_steps++;

#if defined(DFLASH27B_TRIATTENTION_ENABLED)
        // TriAttention KV compression: trigger at specified intervals
        // Only compress when committed tokens exceed kv_budget
        if (g_tria_state.should_compress(committed, committed)) {
            auto t_c0 = std::chrono::steady_clock::now();

            const int n_full_attn = (int)cache_.attn_k.size();
            const int n_head_kv = w_.n_head_kv;
            const int head_dim = 64;  // TriAttention RoPE head_dim (from stats)
            const int tensor_head_dim = w_.n_embd_head_k;  // Actual K tensor head_dim
            const int max_ctx = cache_.max_ctx;

            std::vector<void*> attn_k_ptrs(n_full_attn);
            std::vector<void*> attn_v_ptrs(n_full_attn);
            for (int i = 0; i < n_full_attn; i++) {
                attn_k_ptrs[i] = cache_.attn_k[i]->data;
                attn_v_ptrs[i] = cache_.attn_v[i]->data;
            }

            const int kv_budget = g_tria_state.kv_budget;
            const float budget_ratio = kv_budget > 0 ? (float)kv_budget / (float)std::max(committed, kv_budget) : 0.5f;
            const float keep_ratio = std::max(g_tria_state.min_keep_ratio, budget_ratio);

            int n_kept = 0;
            const bool ok = tria_kv_compress(
                g_tria_state.stats_ptr,
                cache_.tria_k_pre_rope ? cache_.tria_k_pre_rope->data : nullptr,
                attn_k_ptrs.data(),
                attn_v_ptrs.data(),
                n_full_attn,
                n_head_kv,
                head_dim,
                tensor_head_dim,
                max_ctx,
                0,
                committed,
                cfg_.device.gpu,
                keep_ratio,
                &n_kept,
                cache_.kv_k_type,
                cache_.kv_v_type);

            if (ok && n_kept < committed) {
                cache_.cur_pos = n_kept;
                g_tria_state.mark_compressed(n_kept);
                committed = n_kept;
                std::fprintf(stderr, "[TriAttention] Updated cache.cur_pos to %d\n", n_kept);
            }

            auto t_c1 = std::chrono::steady_clock::now();
            const double compress_ms = std::chrono::duration<double>(t_c1 - t_c0).count() * 1000.0;
            std::fprintf(stderr, "[TriAttention] Compression took %.2f ms\n", compress_ms);
        }
#endif
    }

    io.emit(-1);
    return true;
}

int Qwen35Backend::verify_chain(int committed, const int32_t * draft_tok, int q_len) {
    (void)committed; (void)draft_tok; (void)q_len;
    // TODO: Will be implemented when the full spec-decode loop is migrated.
    return 0;
}

int Qwen35Backend::verify_tree(int committed, const DDTree & tree) {
    (void)committed; (void)tree;
    // TODO: Will be implemented when the full spec-decode loop is migrated.
    return 0;
}

}  // namespace dflash27b
