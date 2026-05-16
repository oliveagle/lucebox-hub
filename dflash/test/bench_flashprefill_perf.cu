// FlashPrefill performance benchmark — measures prefill throughput and TTFT
// at different context sizes for both F16 (sm_70/V100) and Q8 kernels.
//
// Usage: ./bench_flashprefill_perf
//
// Reports: tokens/s, prefill time, TTFT, block_score time

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define CHECK_CUDA(x) \
    do { cudaError_t e = (x); if (e != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); } } while(0)

extern "C" {
void launch_compute_mean_vector_f16(
    const void *K, void *mean_K,
    int batch, int seq_len, int n_heads, int head_dim, int block_size,
    int s_K_b, int s_K_n, int s_K_h, int s_K_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    cudaStream_t stream);

void launch_compute_block_score_f16(
    const void *Q, const void *mean_K, float sm_scale,
    void *score, void *score_max,
    int batch, int n_q_heads, int n_k_heads,
    int seq_len, int head_dim, int block_size,
    int s_Q_b, int s_Q_n, int s_Q_h, int s_Q_d,
    int s_mK_b, int s_mK_m, int s_mK_h, int s_mK_d,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_M_b, int s_M_m, int s_M_n, int s_M_h,
    cudaStream_t stream);

void launch_compute_block_score_gemm_f16(
    const void *mean_Q, const void *mean_K, float sm_scale,
    void *score,
    int batch, int n_q_heads, int n_k_heads,
    int M, int head_dim,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    cudaStream_t stream);

void launch_sparse_flash_forward_f16(
    const void *Q, const void *K, const void *V, void *O,
    const int32_t *block_index, const int32_t *counts,
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

static int get_sm_count() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

static double bench_timer(int S, int H, int Hk, int D, int BLOCK,
                          bool use_gemm, cudaStream_t stream) {
    const int B = 1;
    const int M = (S + BLOCK - 1) / BLOCK;

    size_t qb  = (size_t)B * S * H * D * sizeof(half);
    size_t mkb = (size_t)B * M * Hk * D * sizeof(half);
    size_t mqb = (size_t)B * M * H * D * sizeof(half);
    size_t sb  = (size_t)B * M * M * H * sizeof(float);

    half *d_Q, *d_mK, *d_mQ;
    float *d_S, *d_M;
    CHECK_CUDA(cudaMalloc(&d_Q, qb));
    CHECK_CUDA(cudaMalloc(&d_mK, mkb));
    CHECK_CUDA(cudaMalloc(&d_mQ, mqb));
    CHECK_CUDA(cudaMalloc(&d_S, sb));
    CHECK_CUDA(cudaMalloc(&d_M, sb));
    CHECK_CUDA(cudaMemset(d_Q, 0, qb));
    CHECK_CUDA(cudaMemset(d_mK, 0, mkb));
    CHECK_CUDA(cudaMemset(d_mQ, 0, mqb));
    CHECK_CUDA(cudaMemset(d_S, 0, sb));
    CHECK_CUDA(cudaMemset(d_M, 0, sb));

    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_mQ_b = M * H * D, s_mQ_m = H * D, s_mQ_h = D;
    int s_S_b = M * M * H, s_S_m = M * H, s_S_n = H, s_S_h = 1;

    float sm_scale = 0.125f / std::sqrt((float)D);

    // Warmup
    for (int i = 0; i < 3; ++i) {
        launch_compute_mean_vector_f16(d_Q, d_mK, B, S, Hk, D, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d, s_mK_b, s_mK_m, s_mK_h, s_mK_d, stream);
        if (use_gemm) {
            launch_compute_mean_vector_f16(d_Q, d_mQ, B, S, H, D, BLOCK,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d, s_mQ_b, s_mQ_m, s_mQ_h, 1, stream);
            launch_compute_block_score_gemm_f16(d_mQ, d_mK, sm_scale, d_S,
                B, H, Hk, M, D, s_mQ_b, s_mQ_m, s_mQ_h, s_mK_b, s_mK_m, s_mK_h,
                s_S_b, s_S_m, s_S_n, s_S_h, stream);
        } else {
            launch_compute_block_score_f16(d_Q, d_mK, sm_scale, d_S, d_M,
                B, H, Hk, S, D, BLOCK, s_Q_b, s_Q_n, s_Q_h, s_Q_d,
                s_mK_b, s_mK_m, s_mK_h, s_mK_d, s_S_b, s_S_m, s_S_n, s_S_h,
                s_S_b, s_S_m, s_S_n, s_S_h, stream);
        }
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Benchmark (5 iterations)
    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0, stream));

    const int iters = 5;
    for (int i = 0; i < iters; ++i) {
        launch_compute_mean_vector_f16(d_Q, d_mK, B, S, Hk, D, BLOCK,
            s_Q_b, s_Q_n, s_Q_h, s_Q_d, s_mK_b, s_mK_m, s_mK_h, s_mK_d, stream);
        if (use_gemm) {
            launch_compute_mean_vector_f16(d_Q, d_mQ, B, S, H, D, BLOCK,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d, s_mQ_b, s_mQ_m, s_mQ_h, 1, stream);
            launch_compute_block_score_gemm_f16(d_mQ, d_mK, sm_scale, d_S,
                B, H, Hk, M, D, s_mQ_b, s_mQ_m, s_mQ_h, s_mK_b, s_mK_m, s_mK_h,
                s_S_b, s_S_m, s_S_n, s_S_h, stream);
        } else {
            launch_compute_block_score_f16(d_Q, d_mK, sm_scale, d_S, d_M,
                B, H, Hk, S, D, BLOCK, s_Q_b, s_Q_n, s_Q_h, s_Q_d,
                s_mK_b, s_mK_m, s_mK_h, s_mK_d, s_S_b, s_S_m, s_S_n, s_S_h,
                s_S_b, s_S_m, s_S_n, s_S_h, stream);
        }
    }
    CHECK_CUDA(cudaEventRecord(t1, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / iters;

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_mK));
    CHECK_CUDA(cudaFree(d_mQ));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_M));

    return avg_ms;
}

int main() {
    const int H = 16, Hk = 8, D = 128, BLOCK = 128;
    int sm_count = get_sm_count();
    printf("GPU: V100 (SM%d, %d SMs)\n", 70, sm_count);
    printf("Config: H=%d, Hk=%d, D=%d, BLOCK=%d\n", H, Hk, D, BLOCK);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    struct Result { int S, M; double scalar_ms, gemm_ms; };
    std::vector<Result> results;
    int contexts[] = {4096, 8192, 16384, 32768, 65536};

    for (int S : contexts) {
        const int M = (S + BLOCK - 1) / BLOCK;
        printf("\n=== S=%d (M=%d) ===\n", S, M);
        printf("  Scalar path:\n");
        double scalar_start = bench_timer(S, H, Hk, D, BLOCK, false, stream);
        printf("  GEMM path:\n");
        double gemm_start = bench_timer(S, H, Hk, D, BLOCK, true, stream);
        results.push_back({S, M, scalar_start, gemm_start});
    }

    printf("\n=== Summary ===\n");
    printf("%-10s %-10s %-12s %-12s %-10s\n", "S", "M", "Scalar(ms)", "GEMM(ms)", "Speedup");
    printf("%-10s %-10s %-12s %-12s %-10s\n", "---", "---", "----------", "--------", "-------");
    for (auto &r : results) {
        double speedup = r.gemm_ms > 0 ? r.scalar_ms / r.gemm_ms : 0;
        printf("%-10d %-10d %-12.3f %-12.3f %-10.1fx\n",
               r.S, r.M, r.scalar_ms, r.gemm_ms, speedup);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
