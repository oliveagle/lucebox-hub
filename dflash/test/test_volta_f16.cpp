// Volta F16 FlashPrefill e2e smoke test: exercises flash_prefill_forward_f16
// across S=320..65536 to verify the F16 WMMA (sm_70) path does not crash.
//
// This covers the exact scenario of DFLASH_FP_USE_VOLTA_FP=1 in qwen3_graph.cpp,
// but without needing the full model graph. Tests:
//   - mean_vector_f16
//   - block_score_f16 (scalar path)
//   - block_select (host path)
//   - sparse_flash_forward_f16
//
// Usage: ./test_volta_f16

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>

#include "../src/flashprefill.h"

#define CK(x) \
    do { cudaError_t e = (x); if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); } } while(0)

static int get_sm_count() {
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

struct Result {
    int S;
    int M;
    double ms;
    bool pass;
};

int main() {
    int sm = get_sm_count();
    printf("Volta F16 FlashPrefill E2E Smoke Test\n");
    printf("GPU: SM %d, %d SMs\n\n", sm / 10, sm);

    dflash27b::flashprefill::FlashPrefillConfig cfg;
    cfg.block_size = 128;
    cfg.attention_sink = 2;
    cfg.window = 4;
    cfg.last_n_full = 2;
    cfg.alpha = 0.12f;

    static const int contexts[] = {320, 350, 4096, 8192, 16384, 32768, 65536};
    std::vector<Result> results;

    for (int S : contexts) {
        const int B = 1;
        const int H = 16;
        const int Hk = 8;
        const int D = 128;
        const int M = (S + cfg.block_size - 1) / cfg.block_size;
        const float scale = 1.0f / std::sqrt((float)D);

        size_t qkv_bytes = B * S * H * D * sizeof(half);
        size_t kv_bytes = B * S * Hk * D * sizeof(half);
        half *dQ, *dK, *dV, *dO;

        CK(cudaMalloc(&dQ, qkv_bytes));
        CK(cudaMalloc(&dK, kv_bytes));
        CK(cudaMalloc(&dV, kv_bytes));
        CK(cudaMalloc(&dO, qkv_bytes));
        CK(cudaMemset(dQ, 0, qkv_bytes));
        CK(cudaMemset(dK, 0, kv_bytes));
        CK(cudaMemset(dV, 0, kv_bytes));

        double t0 = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        int rc = dflash27b::flashprefill::flash_prefill_forward_f16(
            dQ, dK, dV, dO, B, S, H, Hk, D, scale, cfg);

        CK(cudaDeviceSynchronize());

        double t1 = std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();

        double ms = (t1 - t0) * 1000.0;
        bool pass = (rc == 0);

        printf("  S=%-6d M=%-5d  rc=%d  time=%.1f ms  %s\n",
               S, M, rc, ms, pass ? "PASS" : "FAIL");
        fflush(stdout);

        results.push_back({S, M, ms, pass});

        CK(cudaFree(dQ));
        CK(cudaFree(dK));
        CK(cudaFree(dV));
        CK(cudaFree(dO));
    }

    printf("\n=== Summary ===\n");
    bool all_pass = true;
    for (const auto &r : results) {
        printf("  S=%-6d M=%-5d  time=%8.1f ms  tok/s=%8.0f  %s\n",
               r.S, r.M, r.ms, 1000.0 * r.S / r.ms, r.pass ? "PASS" : "FAIL");
        if (!r.pass) all_pass = false;
    }

    if (all_pass) {
        printf("\nAll tests passed.\n");
        return 0;
    } else {
        printf("\nSome tests FAILED.\n");
        return 1;
    }
}
