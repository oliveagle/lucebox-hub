// Standalone smoke test for V100 GEMM block_score kernel
// Compile: nvcc -arch=sm_70 -o smoke_gemm src/flashprefill_f16_gemm.cu test_gemm_smoke.cu -I src -I deps/llama.cpp/ggml/include

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK_CUDA(x) \
    do { cudaError_t e = (x); if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        return 1; } } while(0)

extern "C" {
void launch_compute_block_score_gemm_f16(
    const void * mean_Q, const void * mean_K, float sm_scale,
    void * score,
    int batch, int n_q_heads, int n_k_heads,
    int M, int head_dim,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    cudaStream_t stream);
}

int main() {
    const int B = 1;
    const int H = 4;        // n_q_heads
    const int Hk = 2;       // n_k_heads
    const int D = 128;
    const int M = 128;      // num blocks (sequence / 128)

    // Allocate mean_Q [B, M, H, D], mean_K [B, M, Hk, D], score [B, M, M, H]
    half * d_mean_Q, * d_mean_K;
    float * d_score;
    size_t mean_Q_size = (size_t)B * M * H * D * sizeof(half);
    size_t mean_K_size = (size_t)B * M * Hk * D * sizeof(half);
    size_t score_size = (size_t)B * M * M * H * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_mean_Q, mean_Q_size));
    CHECK_CUDA(cudaMalloc(&d_mean_K, mean_K_size));
    CHECK_CUDA(cudaMalloc(&d_score, score_size));

    // Initialize to known values: Q = K = 1.0 everywhere
    half * h_mean_Q = new half[B * M * H * D];
    half * h_mean_K = new half[B * M * Hk * D];
    for (int i = 0; i < B * M * H * D; ++i) {
        h_mean_Q[i] = __float2half(1.0f);
    }
    for (int i = 0; i < B * M * Hk * D; ++i) {
        h_mean_K[i] = __float2half(1.0f);
    }

    CHECK_CUDA(cudaMemcpy(d_mean_Q, h_mean_Q, mean_Q_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mean_K, h_mean_K, mean_K_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_score, 0, score_size));

    // Run GEMM kernel
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Strides
    int s_mQ_b = M * H * D, s_mQ_m = H * D, s_mQ_h = D;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D;
    int s_S_b = M * M * H, s_S_m = M * H, s_S_n = H, s_S_h = 1;

    launch_compute_block_score_gemm_f16(
        d_mean_Q, d_mean_K, 0.125f, d_score,
        B, H, Hk, M, D,
        s_mQ_b, s_mQ_m, s_mQ_h,
        s_mK_b, s_mK_m, s_mK_h,
        s_S_b, s_S_m, s_S_n, s_S_h,
        stream);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Copy back and verify
    float * h_score = new float[B * M * M * H];
    CHECK_CUDA(cudaMemcpy(h_score, d_score, score_size, cudaMemcpyDeviceToHost));

    // Verify: GEMM kernel computes S = Q @ K^T * sm_scale * log2(e)
    // For Q = K = 1.0: dot = D = 128, scaled by 0.125 * 1.4427 ≈ 0.1803
    // Result = 128 * 0.1803 ≈ 23.0834
    float expected_val = 0.125f * 1.4426950408889634f * D;  // ≈ 23.0834
    int errors = 0, checked = 0;
    for (int b = 0; b < B; ++b) {
        for (int m = 0; m < M; ++m) {
            for (int n = 0; n <= m; ++n) {  // causal
                for (int h = 0; h < H; ++h) {
                    int idx = ((b * M + m) * M + n) * H + h;
                    float got = h_score[idx];
                    ++checked;
                    if (std::abs(got - expected_val) > 0.5f) {
                        if (errors < 10) {
                            std::fprintf(stderr, "Mismatch at [%d,%d,%d,%d]: got %.2f, expected %.2f\n",
                                        b, m, n, h, got, expected_val);
                        }
                        ++errors;
                    }
                }
            }
        }
    }

    if (errors == 0) {
        std::printf("GEMM kernel smoke test PASSED (M=%d, H=%d, D=%d, checked=%d entries)\n",
                    M, H, D, checked);
        std::printf("  Expected score per element: %.2f\n", expected_val);
    } else {
        std::fprintf(stderr, "GEMM kernel smoke test FAILED: %d errors (of %d checked)\n",
                    errors, checked);
    }

    // Benchmark
    const int warmup = 10, iters = 100;
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    for (int i = 0; i < warmup; ++i) {
        launch_compute_block_score_gemm_f16(
            d_mean_Q, d_mean_K, 0.125f, d_score,
            B, H, Hk, M, D,
            s_mQ_b, s_mQ_m, s_mQ_h,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) {
        launch_compute_block_score_gemm_f16(
            d_mean_Q, d_mean_K, 0.125f, d_score,
            B, H, Hk, M, D,
            s_mQ_b, s_mQ_m, s_mQ_h,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    std::printf("  GEMM kernel time: %.3f ms (%d iterations)\n", ms / iters, iters);

    // FLOPs: 2 * B * H * M * M * D (multiply-add)
    // But only causal upper triangle is computed: ~0.5 * M * M
    float flops = (float)B * H * M * (M + 1) / 2 * D * 2;
    float gflops = flops * iters / (ms * 1e6);
    std::printf("  Effective GFLOPS: %.1f\n", gflops);

    // Cleanup
    delete[] h_mean_Q;
    delete[] h_mean_K;
    delete[] h_score;
    CHECK_CUDA(cudaFree(d_mean_Q));
    CHECK_CUDA(cudaFree(d_mean_K));
    CHECK_CUDA(cudaFree(d_score));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return errors > 0 ? 1 : 0;
}
