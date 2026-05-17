#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>

extern "C" {
void launch_block_select(
    const float *scores, int batch, int n_blocks, int top_k, int n_heads,
    int attention_sink, int window, int last_n_full, float alpha,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    int s_idx_b, int s_idx_m, int s_idx_n, int s_idx_h,
    int s_cnt_b, int s_cnt_m, int s_cnt_h,
    int32_t *block_index, int32_t *counts,
    cudaStream_t stream);
}

int main() {
    const int B = 1, H = 16, BLOCK = 128;
    const int attention_sink = 2, window = 4, last_n_full = 2;
    const float alpha = 0.12f;
    
    int sizes[] = {4096, 8192, 16384, 32768, 65536};
    
    for (int S : sizes) {
        const int M = (S + BLOCK - 1) / BLOCK;
        const int N = M;
        
        size_t sb = B * M * N * H * sizeof(float);
        size_t idxb = B * M * N * H * sizeof(int32_t);
        size_t cntb = B * M * H * sizeof(int32_t);
        
        float *d_S;
        int32_t *d_idx, *d_cnt;
        
        cudaMalloc(&d_S, sb);
        cudaMalloc(&d_idx, idxb);
        cudaMalloc(&d_cnt, cntb);
        
        cudaMemset(d_S, 0, sb);
        cudaMemset(d_idx, 0xff, idxb);
        cudaMemset(d_cnt, 0, cntb);
        
        int s_S_b = M * N * H, s_S_m = N * H, s_S_n = H, s_S_h = 1;
        int s_idx_b = s_S_b, s_idx_m = s_S_m, s_idx_n = s_S_n, s_idx_h = s_S_h;
        int s_cnt_b = M * H, s_cnt_m = H, s_cnt_h = 1;
        
        printf("Testing S=%d (M=%d, grid=%dx%d)...", S, M, B, M, H);
        fflush(stdout);
        
        launch_block_select(d_S, B, M, N, H,
            attention_sink, window, last_n_full, alpha,
            s_S_b, s_S_m, s_S_n, s_S_h,
            s_idx_b, s_idx_m, s_idx_n, s_idx_h,
            s_cnt_b, s_cnt_m, s_cnt_h,
            d_idx, d_cnt, 0);
        
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("LAUNCH ERROR: %s\n", cudaGetErrorString(e));
            return 1;
        }
        cudaDeviceSynchronize();
        e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("SYNC ERROR: %s\n", cudaGetErrorString(e));
            return 1;
        }
        printf("OK\n");
        
        cudaFree(d_S);
        cudaFree(d_idx);
        cudaFree(d_cnt);
    }
    return 0;
}
