#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

extern "C" {
void launch_block_select(
    const float *score,
    int B, int M, int N, int H,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_b, int s_m, int s_n, int s_h,
    int idx_s_b, int idx_s_m, int idx_s_n, int idx_s_h,
    int cnt_s_b, int cnt_s_m, int cnt_s_h,
    int32_t * idx_out, int32_t * cnt_out,
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
}

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::abort(); } } while(0)

int main() {
    const int B = 1, S = 8192, H = 16, Hk = 8, D = 128, BLOCK = 128;
    const int M = (S + BLOCK - 1) / BLOCK;
    const int N = M;
    
    int attention_sink = 2, window = 4, last_n_full = 2;
    float alpha = 0.12f;
    
    size_t qbytes = B * S * H * D * sizeof(half);
    size_t kbytes = B * S * Hk * D * sizeof(half);
    size_t mKbytes = B * M * Hk * D * sizeof(half);
    size_t Sbytes = B * M * N * H * sizeof(float);
    size_t idxbytes = B * M * N * H * sizeof(int32_t);
    size_t cntbytes = B * M * H * sizeof(int32_t);
    
    half *dQ, *dK, *dmK;
    float *dS;
    int32_t *dIdx, *dCnt;
    
    CHECK_CUDA(cudaMalloc(&dQ, qbytes));
    CHECK_CUDA(cudaMalloc(&dK, kbytes));
    CHECK_CUDA(cudaMalloc(&dmK, mKbytes));
    CHECK_CUDA(cudaMalloc(&dS, Sbytes));
    CHECK_CUDA(cudaMalloc(&dIdx, idxbytes));
    CHECK_CUDA(cudaMemset(dIdx, 0xff, idxbytes));  // -1 sentinel
    CHECK_CUDA(cudaMalloc(&dCnt, cntbytes));
    CHECK_CUDA(cudaMemset(dCnt, 0, cntbytes));
    
    half *dO;
    CHECK_CUDA(cudaMalloc(&dO, qbytes));
    CHECK_CUDA(cudaMemset(dO, 0, qbytes));
    
    // Initialize input tensors
    CHECK_CUDA(cudaMemset(dQ, 0, qbytes));
    CHECK_CUDA(cudaMemset(dK, 0, kbytes));
    CHECK_CUDA(cudaMemset(dmK, 0, mKbytes));
    CHECK_CUDA(cudaMemset(dS, 0, Sbytes));
    
    int s_Q_b = S * H * D, s_Q_n = H * D, s_Q_h = D, s_Q_d = 1;
    int s_K_b = S * Hk * D, s_K_n = Hk * D, s_K_h = D, s_K_d = 1;
    int s_mK_b = M * Hk * D, s_mK_m = Hk * D, s_mK_h = D, s_mK_d = 1;
    int s_S_b = M * N * H, s_S_m = N * H, s_S_n = H, s_S_h = 1;
    int s_idx_b = s_S_b, s_idx_m = s_S_m, s_idx_n = s_S_n, s_idx_h = s_S_h;
    int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;
    
    float scale = 0.125f / std::sqrt((float)D);
    
    cudaStream_t stream = 0;
    
    printf("S=%d, M=%d\n", S, M);
    
    // 1. mean_K
    printf("1. mean_K...\n"); fflush(stdout);
    launch_compute_mean_vector_f16(dK, dmK, B, S, Hk, D, BLOCK,
        s_K_b, s_K_n, s_K_h, s_K_d, s_mK_b, s_mK_m, s_mK_h, s_mK_d, stream);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   OK\n");
    
    // 2. block_score
    printf("2. block_score...\n"); fflush(stdout);
    launch_compute_block_score_f16(dQ, dmK, scale, dS, nullptr,
        B, H, Hk, S, D, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_mK_b, s_mK_m, s_mK_h, s_mK_d,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_S_b, s_S_m, s_S_n, s_S_h, stream);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   OK\n");
    
    // 3. block_select
    printf("3. block_select...\n"); fflush(stdout);
    launch_block_select(dS, B, M, N, H,
        attention_sink, window, last_n_full, alpha,
        s_S_b, s_S_m, s_S_n, s_S_h,
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h,
        dIdx, dCnt, stream);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   OK\n");
    
    // 4. sparse_flash_forward
    printf("4. sparse_flash_forward...\n"); fflush(stdout);
    launch_sparse_flash_forward_f16(
        dQ, dK, dK, dO, dIdx, dCnt, scale,
        B, H, Hk, S, D, 64, BLOCK,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_K_b, s_K_n, s_K_h, s_K_d,
        s_Q_b, s_Q_n, s_Q_h, s_Q_d,
        s_idx_b, s_idx_m, s_idx_n, s_idx_h,
        s_cnt_b, s_cnt_m, s_cnt_h, stream);
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("   OK\n");
    
    printf("ALL PASSED\n");
    
    cudaFree(dQ); cudaFree(dK); cudaFree(dmK);
    cudaFree(dS); cudaFree(dIdx); cudaFree(dCnt);
    cudaFree(dO);
    
    return 0;
}
