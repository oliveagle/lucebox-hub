// Standalone HIP benchmark for FlashPrefill BF16 kernels on gfx1151.
// Measures: mean_K, block_score, block_select, sparse_forward.
//
// Usage: just run with no args.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>

#include "device_runtime.h"
#include "flashprefill.h"

#define HIPCK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; \
    } \
} while (0)

static __nv_bfloat16 f2b(float x) { return __float2bfloat16(x); }
static float b2f(__nv_bfloat16 x) { return __bfloat162float(x); }

static int bench(const char *label, int B, int S, int H, int Hk, int D, int BL) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);

    const size_t qkv_size = (size_t)B * S * H * D;
    const size_t k_size   = (size_t)B * S * Hk * D;

    std::vector<__nv_bfloat16> hQ(qkv_size), hK(k_size), hV(k_size);
    for (auto &x : hQ) x = f2b(dist(rng));
    for (auto &x : hK) x = f2b(dist(rng));
    for (auto &x : hV) x = f2b(dist(rng));

    __nv_bfloat16 *dQ, *dK, *dV, *dO;
    HIPCK(cudaMalloc(&dQ, qkv_size * sizeof(__nv_bfloat16)));
    HIPCK(cudaMalloc(&dK, k_size   * sizeof(__nv_bfloat16)));
    HIPCK(cudaMalloc(&dV, k_size   * sizeof(__nv_bfloat16)));
    HIPCK(cudaMalloc(&dO, qkv_size * sizeof(__nv_bfloat16)));

    HIPCK(cudaMemcpy(dQ, hQ.data(), qkv_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    HIPCK(cudaMemcpy(dK, hK.data(), k_size   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    HIPCK(cudaMemcpy(dV, hV.data(), k_size   * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    dflash27b::flashprefill::FlashPrefillConfig cfg;
    cfg.block_size     = BL;
    cfg.attention_sink = 2;
    cfg.window         = 4;
    cfg.last_n_full    = 2;
    cfg.alpha          = 0.12f;

    float scale = 1.0f / std::sqrt((float)D);

    // Warm-up
    dflash27b::flashprefill::flash_prefill_forward_bf16(
        dQ, dK, dV, dO, B, S, H, Hk, D, scale, cfg);
    HIPCK(cudaDeviceSynchronize());

    // Benchmark: 5 iterations
    cudaEvent_t ea, eb;
    HIPCK(cudaEventCreate(&ea));
    HIPCK(cudaEventCreate(&eb));
    HIPCK(cudaEventRecord(ea));
    for (int it = 0; it < 5; ++it) {
        dflash27b::flashprefill::flash_prefill_forward_bf16(
            dQ, dK, dV, dO, B, S, H, Hk, D, scale, cfg);
    }
    HIPCK(cudaEventRecord(eb));
    HIPCK(cudaDeviceSynchronize());
    float ms = 0.0f;
    HIPCK(cudaEventElapsedTime(&ms, ea, eb));

    float per_iter = ms / 5.0f;
    size_t total_flops = (size_t)B * S * H * D * 2; // rough: Q*K + P*V
    float tflops = (total_flops * 2) / (per_iter * 1e9f); // 2 for FMA
    float tokens_s = (double)(B * S) / (per_iter * 1e-3f);

    std::printf("[bench] %s S=%5d H=%3d Hk=%2d D=%3d BL=%3d | %.1f ms/iter | %.2f TFLOP/s | %.0f tok/s\n",
                label, S, H, Hk, D, BL, per_iter, tflops, tokens_s);

    HIPCK(cudaEventDestroy(ea));
    HIPCK(cudaEventDestroy(eb));
    HIPCK(cudaFree(dQ)); HIPCK(cudaFree(dK)); HIPCK(cudaFree(dV)); HIPCK(cudaFree(dO));
    return 0;
}

int main() {
    // Get device info
    int dev;
    HIPCK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    HIPCK(cudaGetDeviceProperties(&prop, dev));
    std::printf("[device] %s (gfx%s), %d CUs, %.0f MHz\n",
                prop.name, prop.gcnArchName,
                prop.multiProcessorCount, prop.clockRate / 1000.0f);

    // Total VRAM
    size_t total_mem, free_mem;
    HIPCK(cudaMemGetInfo(&free_mem, &total_mem));
    std::printf("[device] VRAM: %.1f GB total, %.1f GB free\n",
                (double)total_mem / 1e9, (double)free_mem / 1e9);

    const int B  = 1;
    const int D  = 128;
    const int BL = 128;

    // Prefill benchmarks: varying sequence length, fixed heads
    const int H  = 16;
    const int Hk = 8;

    std::printf("\n─── FlashPrefill BF16 Prefill Benchmark ───\n");
    std::printf("[shape] B=%d H=%d Hk=%d D=%d BL=%d\n\n", B, H, Hk, D, BL);

    int seq_lens[] = {2048, 4096, 8192, 16384, 32768};
    for (int S : seq_lens) {
        // Check memory availability
        size_t needed = (size_t)B * S * H * D * 4; // Q+K+V+O
        if (needed > free_mem * 0.8) {
            std::printf("[bench] S=%d skipped (need %.1f GB, free %.1f GB)\n",
                        S, (double)needed / 1e9, (double)free_mem / 1e9);
            continue;
        }
        if (bench("prefill", B, S, H, Hk, D, BL) != 0) return 1;
        HIPCK(cudaDeviceSynchronize());
    }

    // Also run with per-stage profiling
    std::printf("\n─── Per-stage profiling (S=8192) ───\n");
    setenv("DFLASH_FP_PROFILE", "1", 1);
    setenv("DFLASH_FP_DUMP_COUNTS", "1", 1);
    if (bench("prefill_profile", B, 8192, H, Hk, D, BL) != 0) return 1;
    unsetenv("DFLASH_FP_PROFILE");
    unsetenv("DFLASH_FP_DUMP_COUNTS");

    std::printf("\nDone.\n");
    return 0;
}