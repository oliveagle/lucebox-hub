// FlashPrefill E2E benchmark: block_score + block_select timing.
//
// Tests the key FlashPrefill stages independently:
//   1. mean_vector: compute mean K (and optionally mean Q for GEMM path)
//   2. block_score: compute attention scores for all (q_block, k_block) pairs
//   3. block_select: pick top blocks per query row
//
// Usage:
//   ./bench_flashprefill_e2e           # scalar block_score
//   DFLASH27B_V100_GEMM_SCORE=1 ./bench_flashprefill_e2e  # GEMM block_score
//
// Reports: mean_vector time, block_score time, block_select time, total

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

void launch_block_select(
    const float *scores, int batch, int n_blocks, int top_k, int n_heads,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    int32_t *block_index, int32_t *counts,
    cudaStream_t stream);
}

static int get_sm_count() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    return prop.multiProcessorCount;
}

struct StageTiming {
    double mean_vector_ms = 0;
    double block_score_ms = 0;
    double block_select_ms = 0;
    double total_ms() const { return mean_vector_ms + block_score_ms + block_select_ms; }
};

struct Config {
    int H, Hk, D, BLOCK;
    int window, last_n_full;
    float alpha, attention_sink;
};

static StageTiming bench(int S, const Config & cfg, bool use_gemm, cudaStream_t stream) {
    const int B = 1;
    const int M = (S + cfg.BLOCK - 1) / cfg.BLOCK;
    const float scale = 0.125f / std::sqrt((float)cfg.D);

    // Allocate buffers
    size_t qb  = B * S * cfg.H * cfg.D * sizeof(half);
    size_t kb  = B * S * cfg.Hk * cfg.D * sizeof(half);
    size_t mkb = B * M * cfg.Hk * cfg.D * sizeof(half);
    size_t mqb = B * M * cfg.H * cfg.D * sizeof(half);
    size_t sb  = B * M * M * cfg.H * sizeof(float);
    size_t mb  = B * M * M * cfg.H * sizeof(float);  // score_max
    size_t idxb = B * M * M * cfg.H * sizeof(int32_t);
    size_t cntb = B * M * cfg.H * sizeof(int32_t);

    half *d_Q, *d_K, *d_mK, *d_mQ;
    float *d_S, *d_M;
    int32_t *d_idx, *d_cnt;

    CHECK_CUDA(cudaMalloc(&d_Q, qb));
    CHECK_CUDA(cudaMalloc(&d_K, kb));
    CHECK_CUDA(cudaMalloc(&d_mK, mkb));
    CHECK_CUDA(cudaMalloc(&d_mQ, mqb));
    CHECK_CUDA(cudaMalloc(&d_S, sb));
    CHECK_CUDA(cudaMalloc(&d_M, mb));
    CHECK_CUDA(cudaMalloc(&d_idx, idxb));
    CHECK_CUDA(cudaMalloc(&d_cnt, cntb));

    CHECK_CUDA(cudaMemset(d_Q, 0, qb));
    CHECK_CUDA(cudaMemset(d_K, 0, kb));
    CHECK_CUDA(cudaMemset(d_mK, 0, mkb));
    CHECK_CUDA(cudaMemset(d_mQ, 0, mqb));
    CHECK_CUDA(cudaMemset(d_S, 0, sb));
    CHECK_CUDA(cudaMemset(d_M, 0, mb));
    CHECK_CUDA(cudaMemset(d_idx, 0, idxb));
    CHECK_CUDA(cudaMemset(d_cnt, 0, cntb));

    // Strides
    int s_Q_b = S * cfg.H * cfg.D, s_Q_n = cfg.H * cfg.D, s_Q_h = cfg.D, s_Q_d = 1;
    int s_K_b = S * cfg.Hk * cfg.D, s_K_n = cfg.Hk * cfg.D, s_K_h = cfg.D, s_K_d = 1;
    int s_mK_b = M * cfg.Hk * cfg.D, s_mK_m = cfg.Hk * cfg.D, s_mK_h = cfg.D, s_mK_d = 1;
    int s_mQ_b = M * cfg.H * cfg.D, s_mQ_m = cfg.H * cfg.D, s_mQ_h = cfg.D;
    int s_S_b = M * M * cfg.H, s_S_m = M * cfg.H, s_S_n = cfg.H, s_S_h = 1;
    int s_M_b = s_S_b, s_M_m = s_S_m, s_M_n = s_S_n, s_M_h = s_S_h;
    int s_idx_b = s_S_b, s_idx_m = s_S_m, s_idx_n = s_S_n, s_idx_h = s_S_h;
    int s_cnt_b = M * cfg.H, s_cnt_m = cfg.H, s_cnt_h = 1;

    // For mean_vector K: pass d_K with n_heads=Hk
    // The kernel expects the first tensor to have shape [B, S, n_heads, D]
    // For K we have [B, S, Hk, D], so we pass it directly with Hk as n_heads

    StageTiming t;
    cudaEvent_t t0, t1, t2, t3;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventCreate(&t2));
    CHECK_CUDA(cudaEventCreate(&t3));

    const int iters = 10;

    for (int i = 0; i < iters; ++i) {
        CHECK_CUDA(cudaEventRecord(t0, stream));

        // 1. mean_vector K
        launch_compute_mean_vector_f16(d_K, d_mK, B, S, cfg.Hk, cfg.D, cfg.BLOCK,
            s_K_b, s_K_n, s_K_h, s_K_d, s_mK_b, s_mK_m, s_mK_h, s_mK_d, stream);

        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "mean_vector_K launch error: %s\n", cudaGetErrorString(e)); std::abort(); }
        cudaStreamSynchronize(stream);
        e = cudaGetLastError();
        if (e != cudaSuccess) { fprintf(stderr, "mean_vector_K sync error: %s\n", cudaGetErrorString(e)); std::abort(); }

        if (use_gemm) {
            // 1b. mean_vector Q (GEMM path only)
            launch_compute_mean_vector_f16(d_Q, d_mQ, B, S, cfg.H, cfg.D, cfg.BLOCK,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d, s_mQ_b, s_mQ_m, s_mQ_h, 1, stream);
            {
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) { fprintf(stderr, "mean_vector_Q launch error: %s\n", cudaGetErrorString(e)); std::abort(); }
            }
            cudaStreamSynchronize(stream);
            {
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) { fprintf(stderr, "mean_vector_Q sync error: %s\n", cudaGetErrorString(e)); std::abort(); }
            }
        }

        CHECK_CUDA(cudaEventRecord(t1, stream));

        // 2. block_score
        if (use_gemm) {
            launch_compute_block_score_gemm_f16(d_mQ, d_mK, scale, d_S,
                B, cfg.H, cfg.Hk, M, cfg.D,
                s_mQ_b, s_mQ_m, s_mQ_h, s_mK_b, s_mK_m, s_mK_h,
                s_S_b, s_S_m, s_S_n, s_S_h, stream);
        } else {
            launch_compute_block_score_f16(d_Q, d_mK, scale, d_S, d_M,
                B, cfg.H, cfg.Hk, S, cfg.D, cfg.BLOCK,
                s_Q_b, s_Q_n, s_Q_h, s_Q_d,
                s_mK_b, s_mK_m, s_mK_h, s_mK_d,
                s_S_b, s_S_m, s_S_n, s_S_h,
                s_M_b, s_M_m, s_M_n, s_M_h, stream);
        }
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "block_score launch error: %s\n", cudaGetErrorString(e)); std::abort(); }
        }
        cudaStreamSynchronize(stream);
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "block_score sync error: %s\n", cudaGetErrorString(e)); std::abort(); }
        }

        CHECK_CUDA(cudaEventRecord(t2, stream));

        // 3. block_select
        launch_block_select((const float*)d_S, B, M, M, cfg.H,
            cfg.attention_sink, cfg.window, cfg.last_n_full, cfg.alpha,
            s_S_b, s_S_m, s_S_n, s_S_h, s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h, d_idx, d_cnt, stream);
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "block_select launch error: %s\n", cudaGetErrorString(e)); std::abort(); }
        }
        cudaStreamSynchronize(stream);
        {
            cudaError_t e = cudaGetLastError();
            if (e != cudaSuccess) { fprintf(stderr, "block_select sync error: %s\n", cudaGetErrorString(e)); std::abort(); }
        }

        CHECK_CUDA(cudaEventRecord(t3, stream));
        CHECK_CUDA(cudaStreamSynchronize(stream));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1)); t.mean_vector_ms += ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t1, t2)); t.block_score_ms += ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, t2, t3)); t.block_select_ms += ms;
    }

    t.mean_vector_ms /= iters;
    t.block_score_ms /= iters;
    t.block_select_ms /= iters;

    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    CHECK_CUDA(cudaEventDestroy(t2));
    CHECK_CUDA(cudaEventDestroy(t3));

    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_mK));
    CHECK_CUDA(cudaFree(d_mQ));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_M));
    CHECK_CUDA(cudaFree(d_idx));
    CHECK_CUDA(cudaFree(d_cnt));

    return t;
}

int main() {
    Config cfg{16, 8, 128, 128, 4, 2, 0.12f, 2.0f};
    int sm_count = get_sm_count();

    printf("FlashPrefill E2E Benchmark (block_score + block_select)\n");
    printf("GPU: V100 (SM70, %d SMs)\n", sm_count);
    printf("Config: H=%d, Hk=%d, D=%d, BLOCK=%d\n", cfg.H, cfg.Hk, cfg.D, cfg.BLOCK);
    printf("block_select: window=%d, last_n=%d, alpha=%.2f\n\n",
           cfg.window, cfg.last_n_full, cfg.alpha);

    static const bool use_gemm = (std::getenv("DFLASH27B_V100_GEMM_SCORE") != nullptr);
    printf("Block score path: %s\n\n", use_gemm ? "GEMM (Tensor Core)" : "Scalar");
    fflush(stdout);

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    struct Result { int S, M; StageTiming scalar_t, gemm_t; };
    std::vector<Result> results;
    int contexts[] = {320, 350, 4096, 8192, 16384, 32768, 65536};

    for (int S : contexts) {
        const int M = (S + cfg.BLOCK - 1) / cfg.BLOCK;
        printf("=== S=%d (M=%d blocks) ===\n", S, M);
        fflush(stdout);

        StageTiming scalar_t = bench(S, cfg, false, stream);
        StageTiming gemm_t  = bench(S, cfg, true,  stream);

        results.push_back({S, M, scalar_t, gemm_t});

        printf("  Scalar:\n");
        printf("    mean_vector:  %8.2f ms\n", scalar_t.mean_vector_ms);
        printf("    block_score:  %8.2f ms\n", scalar_t.block_score_ms);
        printf("    block_select: %8.2f ms\n", scalar_t.block_select_ms);
        printf("    TOTAL:        %8.2f ms\n", scalar_t.total_ms());

        printf("  GEMM:\n");
        printf("    mean_vector:  %8.2f ms  (scalar was %.2f ms)\n",
               gemm_t.mean_vector_ms, scalar_t.mean_vector_ms);
        printf("    block_score: %8.2f ms  (scalar was %.2f ms, speedup: %.1fx)\n",
               gemm_t.block_score_ms, scalar_t.block_score_ms,
               scalar_t.block_score_ms / gemm_t.block_score_ms);
        printf("    block_select: %8.2f ms\n", gemm_t.block_select_ms);
        printf("    TOTAL:        %8.2f ms  (scalar was %.2f ms, speedup: %.2fx)\n",
               gemm_t.total_ms(), scalar_t.total_ms(),
               scalar_t.total_ms() / gemm_t.total_ms());
        printf("    tok/s: %8.0f\n\n", 1000.0 * S / gemm_t.total_ms());
    }

    printf("=== Summary ===\n");
    printf("%-8s %-6s %-10s %-10s %-10s %-10s %-10s\n",
           "S", "M", "S-total", "S-score", "G-total", "G-score", "Speedup");
    printf("%-8s %-6s %-10s %-10s %-10s %-10s %-10s\n",
           "---", "---", "--------", "--------", "--------", "--------", "-------");
    for (auto & r : results) {
        double speedup = r.gemm_t.total_ms() > 0
                        ? r.scalar_t.total_ms() / r.gemm_t.total_ms() : 0;
        printf("%-8d %-6d %-10.2f %-10.2f %-10.2f %-10.2f %-10.2fx\n",
               r.S, r.M,
               r.scalar_t.total_ms(), r.scalar_t.block_score_ms,
               r.gemm_t.total_ms(), r.gemm_t.block_score_ms,
               speedup);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}