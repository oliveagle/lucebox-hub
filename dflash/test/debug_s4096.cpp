// Quick debug test for S=4096 only
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

int main() {
    int S = 4096;
    dflash27b::flashprefill::FlashPrefillConfig cfg;
    cfg.block_size = 128;
    cfg.attention_sink = 2;
    cfg.window = 4;
    cfg.last_n_full = 2;
    cfg.alpha = 0.12f;

    const int B = 1, H = 16, Hk = 8, D = 128;
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

    int rc = dflash27b::flashprefill::flash_prefill_forward_f16(
        dQ, dK, dV, dO, B, S, H, Hk, D, scale, cfg);

    CK(cudaDeviceSynchronize());

    printf("S=%d rc=%d\n", S, rc);
    if (rc == 0) printf("PASS\n"); else printf("FAIL\n");

    CK(cudaFree(dQ));
    CK(cudaFree(dK));
    CK(cudaFree(dV));
    CK(cudaFree(dO));
    return 0;
}
