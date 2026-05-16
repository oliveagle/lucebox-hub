// FlashPrefill V100 benchmarks - individual kernel tests

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CHECK_CUDA(x) \
    do { cudaError_t e = (x); if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); } } while(0)

extern "C" {
void launch_compute_block_score_f16(
    const void * Q, const void * mean_K, float sm_scale,
    void * score, void * score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

void launch_compute_block_score_gemm_f16(
    const void * mean_Q, const void * mean_K, float sm_scale,
    void * score,
    int batch, int n_q_heads, int n_k_heads,
    int M, int head_dim,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    cudaStream_t stream);

void launch_compute_mean_vector_f16(
    const void * K, void * mean_K,
    int batch, int seq_len, int n_kv_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

void launch_sparse_flash_forward_f16(
    const void * Q, const void * K, const void * V, void * O,
    const int32_t * block_index, const int32_t * counts,
    float scale,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int q_tile, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_V_b, int s_V_n, int s_V_h, int s_V_d,
    int s_O_b, int s_O_n, int s_O_h, int s_O_d,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    cudaStream_t stream);
}

void benchmark_score_kernel(int S, int H, int Hk, int D, int BLOCK, cudaStream_t stream) {
    const int M = (S + BLOCK - 1) / BLOCK;
    printf("\n--- block_score benchmark: S=%d, H=%d, Hk=%d, D=%d, M=%d ---\n", S, H, Hk, D, M);

    // Allocate Q[B,S,H,D], mean_K[B,M,Hk,D], score[B,M,M,H], score_max[B,M,M,H]
    const int B = 1;
    half * d_Q, * d_mK;
    float * d_S, * d_SM;
    size_t q_size = (size_t)B * S * H * D * sizeof(half);
    size_t mk_size = (size_t)B * M * Hk * D * sizeof(half);
    size_t s_size = (size_t)B * M * M * H * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_Q, q_size));
    CHECK_CUDA(cudaMalloc(&d_mK, mk_size));
    CHECK_CUDA(cudaMalloc(&d_S, s_size));
    CHECK_CUDA(cudaMalloc(&d_SM, s_size));

    CHECK_CUDA(cudaMemset(d_Q, 0, q_size));
    CHECK_CUDA(cudaMemset(d_mK, 0, mk_size));
    CHECK_CUDA(cudaMemset(d_S, 0, s_size));
    CHECK_CUDA(cudaMemset(d_SM, 0, s_size));

    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * M * H, s_S_m = M * H, s_S_n = H, s_S_h = 1;

    // Warmup
    for (int i = 0; i < 5; ++i) {
        launch_compute_block_score_f16(d_Q, d_mK, 0.125f / std::sqrt((float)D),
            d_S, d_SM, B, H, Hk, S, D, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Benchmark
    const int iters = 20;
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));

    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) {
        launch_compute_block_score_f16(d_Q, d_mK, 0.125f / std::sqrt((float)D),
            d_S, d_SM, B, H, Hk, S, D, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d,
            s_mK_b, s_mK_m, s_mK_h, s_mK_d,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    float avg_ms = ms / iters;
    printf("  Score (scalar) kernel time: %.3f ms\n", avg_ms);

    // Benchmark GEMM kernel
    half * d_mQ;
    size_t mq_size = (size_t)B * M * H * D * sizeof(half);
    CHECK_CUDA(cudaMalloc(&d_mQ, mq_size));
    CHECK_CUDA(cudaMemset(d_mQ, 0, mq_size));

    launch_compute_mean_vector_f16(d_Q, d_mQ, B, S, H, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        M * H * D, H * D, D, 1, stream);
    CHECK_CUDA(cudaMemset(d_S, 0, s_size));

    // Warmup
    for (int i = 0; i < 5; ++i) {
        launch_compute_block_score_gemm_f16(d_mQ, d_mK, 0.125f / std::sqrt((float)D),
            d_S, B, H, Hk, M, D,
            M * H * D, H * D, D,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventRecord(t0, stream));
    for (int i = 0; i < iters; ++i) {
        launch_compute_block_score_gemm_f16(d_mQ, d_mK, 0.125f / std::sqrt((float)D),
            d_S, B, H, Hk, M, D,
            M * H * D, H * D, D,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h,
            stream);
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    avg_ms = ms / iters;
    printf("  Score (GEMM)    kernel time: %.3f ms\n", avg_ms);

    // Speedup
    float gflops = (float)B * H * M * (M + 1) / 2 * D * 2 * iters / (ms * 1e6);
    printf("  GEMM GFLOPS: %.1f\n", gflops);

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_mK));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_SM));
    CHECK_CUDA(cudaFree(d_mQ));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
}

int main() {
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    const int H = 16, Hk = 8, D = 128, BLOCK = 128;

    benchmark_score_kernel(4096, H, Hk, D, BLOCK, stream);
    benchmark_score_kernel(8192, H, Hk, D, BLOCK, stream);
    benchmark_score_kernel(16384, H, Hk, D, BLOCK, stream);
    benchmark_score_kernel(32768, H, Hk, D, BLOCK, stream);

    printf("\n=== All benchmarks complete ===\n");

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
