// TTFT (time-to-first-token) bench for Laguna in dflash. Loads Laguna Q4_K_M,
// allocates a cache sized for the longest prefill, then for each context
// length: builds a prefill graph for N tokens (synthetic input), runs it on
// CUDA, measures wall time. Reports TTFT @ each length.
//
// Usage:
//   bench_laguna_ttft <laguna.gguf> ["4096,16384,32768"]
//
// The synthetic input uses token id 1972 repeated N times (avoids BOS
// special-casing; any non-special id works, the bench measures wall time not
// generation quality).
//
// On RTX 3090 24 GB the practical ceiling without KV bit-reduction:
//   Q8_0 KV  + 18.77 GiB weights -> ~32K context
//   For 64K+ need Q4_0 KV (planned, not in this bench).

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-alloc.h"

using namespace dflash27b;

static std::vector<int> parse_csv(const std::string & s, std::vector<int> dflt) {
    if (s.empty()) return dflt;
    std::vector<int> out;
    size_t start = 0;
    while (start < s.size()) {
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <laguna.gguf> [\"4096,16384,32768\"]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const std::string lens_csv = (argc >= 3) ? argv[2] : "";
    std::vector<int> ctx_lens = parse_csv(lens_csv, {1024, 4096, 16384});
    int max_len = 0;
    for (int n : ctx_lens) if (n > max_len) max_len = n;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    if (!create_laguna_target_cache(w, max_len, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }
    std::printf("[bench] cache max_ctx=%d  KV bytes/layer ~ %.1f MiB\n",
                max_len, (2.0 * w.head_dim * max_len * w.n_head_kv) / (1024.0 * 1024.0));

    const int32_t fake_tok = 1972;  // "hello" or whatever; just a non-special id

    for (int N : ctx_lens) {
        if (N > max_len) { std::printf("[bench] skip N=%d > max_len=%d\n", N, max_len); continue; }
        reset_laguna_target_cache(cache);

        // Build embedding tensor for N tokens: dequantize from CPU embedder.
        std::vector<int32_t> ids((size_t)N, fake_tok);
        std::vector<float> embed_f32((size_t)N * w.n_embd);
        if (!w.embedder.embed(ids.data(), N, embed_f32.data())) {
            std::fprintf(stderr, "embed failed at N=%d\n", N);
            continue;
        }

        // Per-call ggml context for graph tensors.
        ggml_init_params ip{};
        ip.mem_size = ggml_tensor_overhead() * 16384 + ggml_graph_overhead() + 16 * 1024 * 1024;
        ip.no_alloc = true;
        ggml_context * ctx = ggml_init(ip);
        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 16384, false);

        ggml_tensor * inp_embed = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, w.n_embd, N, 1);
        ggml_set_name(inp_embed, "inp_embed");
        ggml_set_input(inp_embed);

        ggml_tensor * positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, N);
        ggml_set_name(positions, "positions");
        ggml_set_input(positions);

        // Causal mask. Skip if DFLASH_NO_MASK set (semantically wrong but useful
        // to isolate FA kernel issues). ggml_flash_attn_ext expects mask shape:
        //   ne[0] = n_kv (no padding required here)
        //   ne[1] = n_tokens padded to GGML_KQ_MASK_PAD (64)
        // F16 dtype. Row-major: row stride = ne[0] elements.
        const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);
        // llama.cpp/build_attn convention: mask shape [GGML_PAD(n_kv, 64), n_tokens]
        // F16. ne[0] = padded n_kv (fast), ne[1] = n_tokens.
        const int pad = 64;
        const int kv_pad = ((N + pad - 1) / pad) * pad;
        ggml_tensor * mask = nullptr;
        if (!no_mask && N > 1) {
            mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, kv_pad, N);
            ggml_set_name(mask, "causal_mask");
            ggml_set_input(mask);
        }

        LagunaGraphInputs gi{};
        gi.inp_embed     = inp_embed;
        gi.positions     = positions;
        gi.attn_mask     = mask;  // may be nullptr
        gi.n_tokens      = N;
        gi.kv_start      = 0;
        gi.output_logits = true;

        LagunaGraphOutputs go = build_laguna_graph(ctx, gf, w, cache, gi);
        if (!go.logits) { std::fprintf(stderr, "no logits N=%d\n", N); ggml_free(ctx); continue; }

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            std::fprintf(stderr, "gallocr_alloc N=%d failed\n", N);
            ggml_gallocr_free(galloc); ggml_free(ctx); continue;
        }

        // Upload inputs.
        ggml_backend_tensor_set(inp_embed, embed_f32.data(), 0, embed_f32.size() * sizeof(float));
        std::vector<int32_t> pos(N);
        for (int i = 0; i < N; ++i) pos[i] = i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
        if (mask) {
            // [kv_pad, N] F16. Row stride = kv_pad. Cols beyond N (kv_pad..) =
            // -inf (no real K/V there). Within real cols [0..N), causal: j > t = -inf.
            std::vector<uint16_t> mask_buf((size_t)kv_pad * N, 0);
            const uint16_t F16_NEG_INF = 0xFC00;
            const bool zero_mask = (std::getenv("DFLASH_MASK_ZERO") != nullptr);
            for (int t = 0; t < N; ++t) {
                for (int j = 0; j < kv_pad; ++j) {
                    bool masked = false;
                    if (j >= N) masked = true;
                    else if (!zero_mask && j > t) masked = true;
                    mask_buf[(size_t)t * kv_pad + j] = masked ? F16_NEG_INF : 0;
                }
            }
            ggml_backend_tensor_set(mask, mask_buf.data(), 0, mask_buf.size() * sizeof(uint16_t));
        }

        // Warm-up kernels (CUDA JIT, gallocr alloc, etc).
        ggml_backend_graph_compute(backend, gf);

        // Time the actual prefill.
        auto t0 = std::chrono::steady_clock::now();
        ggml_backend_synchronize(backend);
        auto t1 = std::chrono::steady_clock::now();
        ggml_status st = ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
        auto t2 = std::chrono::steady_clock::now();
        if (st != GGML_STATUS_SUCCESS) {
            std::fprintf(stderr, "compute N=%d failed status=%d\n", N, (int)st);
            ggml_gallocr_free(galloc); ggml_free(ctx); continue;
        }
        const double sync0 = std::chrono::duration<double>(t1 - t0).count();
        const double pf_s  = std::chrono::duration<double>(t2 - t1).count();

        // Argmax of last-position logits as a sanity check.
        const int64_t vocab = go.logits->ne[0];
        std::vector<float> logits_last((size_t)vocab);
        ggml_backend_tensor_get(go.logits, logits_last.data(),
                                 (size_t)(N - 1) * vocab * sizeof(float),
                                 logits_last.size() * sizeof(float));
        int best = 0; float bv = logits_last[0]; int n_inf = 0, n_nan = 0;
        for (int i = 0; i < (int)vocab; ++i) {
            float v = logits_last[i];
            if (std::isnan(v)) ++n_nan;
            if (std::isinf(v)) ++n_inf;
            if (v > bv) { bv = v; best = i; }
        }

        std::printf("[bench] N=%6d  TTFT=%8.3f s  (%6.1f tok/s)  argmax=%d  logit=%.3f  nan=%d inf=%d\n",
                    N, pf_s, N / std::max(1e-9, pf_s), best, bv, n_nan, n_inf);
        (void)sync0;

        ggml_gallocr_free(galloc);
        ggml_free(ctx);
    }

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}
