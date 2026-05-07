// Laguna daemon library implementation. Hosts the Poolside Laguna-XS.2
// target on a fresh CUDA backend and services stdin commands. See
// laguna_daemon.h for the protocol overview.
//
// Invoked from two places:
//   - test/test_dflash.cpp: arch dispatch — when the GGUF reports
//     `general.architecture == "laguna"`, main() builds a LagunaDaemonArgs
//     and hands off here, bypassing the qwen35 + DFlash + DDTree code path.
//   - test/test_laguna_daemon.cpp: legacy thin wrapper kept for the NIAH
//     driver (scripts/laguna_pflash_niah.py) which spawns the binary
//     directly with `--max-ctx`/`--kv`/`--chunk`/`--stream-fd` flags.

#include "laguna_daemon.h"

#include "laguna_internal.h"
#include "internal.h"
#include "dflash27b.h"
#include "sampler.h"  // shared CPU sampler chain (SamplerCfg /
                      // sample_logits / parse_sampler_token), defined once
                      // in src/sampler.cpp and reused by test_dflash.cpp.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "ggml-backend.h"
#include "ggml-cuda.h"

#ifndef _WIN32
#include <unistd.h>
#else
#include <io.h>
#include <fcntl.h>
#define ssize_t long
#endif

namespace dflash27b {

namespace {

// laguna_serve.py + laguna_pflash_niah.py write prompts as a uint32 length
// prefix followed by N int32 token IDs. Used by the legacy `generate` path.
std::vector<int32_t> read_counted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    uint32_t n = 0;
    f.read(reinterpret_cast<char *>(&n), sizeof(n));
    if (!f) return {};
    std::vector<int32_t> ids((size_t)n);
    if (n > 0) {
        f.read(reinterpret_cast<char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

// scripts/server.py writes prompts as a raw int32 stream (no length prefix);
// the file size implies the token count. Used by the bare-prompt path so the
// daemon stays drop-in for the qwen35 server.py protocol.
std::vector<int32_t> read_uncounted_i32(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const auto sz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<int32_t> ids(sz / sizeof(int32_t));
    if (!ids.empty()) {
        f.read(reinterpret_cast<char *>(ids.data()),
               (std::streamsize)ids.size() * sizeof(int32_t));
        if (!f) return {};
    }
    return ids;
}

bool write_counted_i32(const std::string & path, const std::vector<int32_t> & ids) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)ids.size();
    f.write(reinterpret_cast<const char *>(&n), sizeof(n));
    if (n > 0) f.write(reinterpret_cast<const char *>(ids.data()), (std::streamsize)ids.size() * sizeof(int32_t));
    return (bool)f;
}

// laguna_step lives in src/laguna_target_graph.cpp as a public helper so the
// daemon, benches, and any future caller share one forward-step
// implementation. We just call it from the daemon loop below.

}  // namespace


int run_laguna_daemon(const LagunaDaemonArgs & args) {
    const bool no_mask = (std::getenv("DFLASH_NO_MASK") != nullptr);

    int stream_fd = args.stream_fd;
    auto emit_int32 = [&](int32_t v) {
        if (stream_fd < 0) return;
        const int32_t w = v;
        ssize_t n = ::write(stream_fd, &w, sizeof(w));
        (void)n;
    };

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) { std::fprintf(stderr, "cuda init failed\n"); return 1; }

    LagunaTargetWeights w;
    if (!load_target_gguf_laguna(args.target_path, backend, w)) {
        std::fprintf(stderr, "load failed: %s\n", dflash27b_last_error());
        ggml_backend_free(backend); return 1;
    }

    LagunaTargetCache cache;
    cache.kv_k_type = args.kv_type;
    cache.kv_v_type = args.kv_type;
    if (!create_laguna_target_cache(w, args.max_ctx, backend, cache)) {
        std::fprintf(stderr, "cache failed: %s\n", dflash27b_last_error());
        free_laguna_target_weights(w); ggml_backend_free(backend); return 1;
    }

    std::printf("[laguna-daemon] ready vocab=%lld eos=%d eot=%d max_ctx=%d kv=%s chunk=%d\n",
                (long long)w.embedder.n_vocab, w.eos_id, w.eos_chat_id, args.max_ctx,
                ggml_type_name(args.kv_type), args.chunk);
    std::fflush(stdout);

    std::mt19937_64 sampler_rng{std::random_device{}()};

    auto run_prompt = [&](const std::vector<int32_t> & prompt,
                          int n_gen,
                          const SamplerCfg & sampler,
                          bool do_sample,
                          bool stream,
                          double & pf_s_out,
                          double & g_s_out,
                          std::vector<int32_t> & generated_out) -> const char * {
        const int N = (int)prompt.size();
        if (N + n_gen > args.max_ctx) return "overflow";

        reset_laguna_target_cache(cache);

        std::vector<float> embed_pf((size_t)N * w.n_embd);
        if (!w.embedder.embed(prompt.data(), N, embed_pf.data())) return "embed_prefill";

        auto t_pf0 = std::chrono::steady_clock::now();
        std::vector<float> last_logits;
        bool ok = true;
        const int n_chunks = (N + args.chunk - 1) / args.chunk;
        for (int c = 0; c < n_chunks && ok; ++c) {
            const int kv_start = c * args.chunk;
            const int n_tok    = std::min(args.chunk, N - c * args.chunk);
            ok = laguna_step(backend, w, cache,
                              embed_pf.data() + (size_t)kv_start * w.n_embd,
                              n_tok, kv_start, no_mask, last_logits);
        }
        if (!ok) return "prefill";
        auto t_pf1 = std::chrono::steady_clock::now();
        pf_s_out = std::chrono::duration<double>(t_pf1 - t_pf0).count();

        auto argmax = [](const std::vector<float> & ll) {
            int best = 0; float bv = ll[0];
            for (size_t i = 1; i < ll.size(); ++i)
                if (ll[i] > bv) { bv = ll[i]; best = (int)i; }
            return best;
        };

        std::vector<int32_t> history;
        history.reserve((size_t)N + (size_t)n_gen);
        history.insert(history.end(), prompt.begin(), prompt.end());

        auto pick = [&](const std::vector<float> & ll) -> int {
            return do_sample
                ? sample_logits(ll.data(), (int)ll.size(), sampler, history, sampler_rng)
                : argmax(ll);
        };

        int next_tok = pick(last_logits);
        generated_out.clear();
        generated_out.reserve(n_gen);

        std::vector<float> embed_step((size_t)w.n_embd);
        auto t_g0 = std::chrono::steady_clock::now();
        for (int s = 0; s < n_gen; ++s) {
            if (next_tok == w.eos_id || next_tok == w.eos_chat_id) break;
            generated_out.push_back(next_tok);
            history.push_back(next_tok);
            if (stream) emit_int32(next_tok);
            if (!w.embedder.embed(&next_tok, 1, embed_step.data())) { ok = false; break; }
            std::vector<float> step_logits;
            if (!laguna_step(backend, w, cache, embed_step.data(), 1,
                              cache.cur_pos, no_mask, step_logits)) { ok = false; break; }
            next_tok = pick(step_logits);
        }
        auto t_g1 = std::chrono::steady_clock::now();
        g_s_out = std::chrono::duration<double>(t_g1 - t_g0).count();

        if (stream) emit_int32(-1);
        return ok ? nullptr : "decode";
    };

    auto looks_like_path = [](const std::string & s) {
        if (s.empty()) return false;
        if (s[0] == '/' || s[0] == '.') return true;
        return s.find('/') != std::string::npos;
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line == "quit" || line == "exit") break;

        SamplerCfg sampler{};
        const bool have_sampler = parse_sampler_token(line, sampler);
        if (have_sampler && sampler.seed != 0) sampler_rng.seed(sampler.seed);
        const bool do_sample = have_sampler && sampler.temp > 0.0f;
        std::istringstream iss(line);
        std::string cmd;
        iss >> cmd;

        if (cmd == "generate") {
            std::string in_path, out_path;
            int n_gen = 0;
            iss >> in_path >> n_gen >> out_path;
            if (in_path.empty() || out_path.empty() || n_gen <= 0) {
                std::fprintf(stderr, "[laguna-daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_counted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout); continue;
            }
            double pf_s = 0.0, g_s = 0.0;
            std::vector<int32_t> generated;
            const char * err = run_prompt(prompt, n_gen, sampler, do_sample,
                                          /*stream=*/false, pf_s, g_s, generated);
            if (err) {
                std::printf("err %s\n", err); std::fflush(stdout);
                continue;
            }
            if (!write_counted_i32(out_path, generated)) {
                std::printf("err write_out\n"); std::fflush(stdout); continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f out=%s\n",
                        (int)prompt.size(), generated.size(), pf_s, g_s,
                        generated.size() / std::max(1e-9, g_s), out_path.c_str());
            std::fflush(stdout);
            continue;
        }

        if (looks_like_path(cmd)) {
            const std::string & in_path = cmd;
            int n_gen = 0;
            iss >> n_gen;
            if (n_gen <= 0) {
                std::fprintf(stderr, "[laguna-daemon] bad: %s\n", line.c_str());
                std::printf("err bad_args\n"); std::fflush(stdout);
                emit_int32(-1);
                continue;
            }
            if (stream_fd < 0) {
                std::fprintf(stderr, "[laguna-daemon] bare-prompt requires --stream-fd\n");
                std::printf("err no_stream_fd\n"); std::fflush(stdout);
                continue;
            }
            auto prompt = read_uncounted_i32(in_path);
            if (prompt.empty()) {
                std::printf("err empty_prompt\n"); std::fflush(stdout);
                emit_int32(-1);
                continue;
            }
            double pf_s = 0.0, g_s = 0.0;
            std::vector<int32_t> generated;
            const char * err = run_prompt(prompt, n_gen, sampler, do_sample,
                                          /*stream=*/true, pf_s, g_s, generated);
            if (err) {
                emit_int32(-1);
                std::printf("err %s\n", err); std::fflush(stdout);
                continue;
            }
            std::printf("ok N=%d gen=%zu prefill_s=%.3f decode_s=%.3f decode_tok_s=%.1f stream_fd=%d\n",
                        (int)prompt.size(), generated.size(), pf_s, g_s,
                        generated.size() / std::max(1e-9, g_s), stream_fd);
            std::fflush(stdout);
            continue;
        }

        std::fprintf(stderr, "[laguna-daemon] unknown cmd: %s\n", line.c_str());
        std::printf("err unknown_command\n"); std::fflush(stdout);
        emit_int32(-1);
    }

    free_laguna_target_cache(cache);
    free_laguna_target_weights(w);
    ggml_backend_free(backend);
    return 0;
}

}  // namespace dflash27b
