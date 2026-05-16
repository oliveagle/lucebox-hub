// F16 WMMA GEMM block_score kernel for V100 (sm_70).
// Optimized version with 512 threads (16 warps).

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

namespace dflash27b {
namespace flashprefill {

template <int Q_TILE, int K_TILE, int D_HEAD>
__global__ void __launch_bounds__(512) compute_block_score_gemm_kernel_f16(
    const half * __restrict__ mean_Q,
    const half * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,
    int B, int M, int n_q_heads, int n_k_heads,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h)
{
    using namespace nvcuda;

    constexpr int MMA_M = 32, MMA_N = 8, MMA_K = 16;
    constexpr int NDK = D_HEAD / MMA_K;
    constexpr int NNK = K_TILE / MMA_N;

    const int b = blockIdx.y / n_q_heads;
    const int qh = blockIdx.y % n_q_heads;
    const int kh = qh * n_k_heads / n_q_heads;
    if (b >= B) return;

    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int mq = wid / NNK;
    const int nk = wid % NNK;

    extern __shared__ unsigned char smem_raw[];
    half * Q_sh = reinterpret_cast<half*>(smem_raw);
    half * K_sh = Q_sh + (size_t)Q_TILE * D_HEAD;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> S_frag;
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> Af;
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> Bf;

    const float scale = sm_scale * 1.4426950408889634f;

    const int n_qt = (M + Q_TILE - 1) / Q_TILE;
    for (int qt = 0; qt < n_qt; ++qt) {
        const int q_base = qt * Q_TILE;
        const int q_limit = min(M - q_base, Q_TILE);

        for (int idx = threadIdx.x; idx < Q_TILE * D_HEAD; idx += blockDim.x) {
            const int row = idx / D_HEAD;
            const int col = idx % D_HEAD;
            if (row < q_limit) {
                Q_sh[idx] = mean_Q[
                    (size_t)b * s_mQ_b + (size_t)(q_base + row) * s_mQ_m
                    + (size_t)qh * s_mQ_h + col];
            } else {
                Q_sh[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int kt = 0; kt <= qt; ++kt) {
            const int k_base = kt * K_TILE;
            const int k_limit = min(M - k_base, K_TILE);

            for (int idx = threadIdx.x; idx < K_TILE * D_HEAD; idx += blockDim.x) {
                const int row = idx / D_HEAD;
                const int col = idx % D_HEAD;
                if (row < k_limit) {
                    K_sh[(size_t)row * D_HEAD + col] = mean_K[
                        (size_t)b * s_mK_b + (size_t)(k_base + row) * s_mK_m
                        + (size_t)kh * s_mK_h + col];
                } else {
                    K_sh[(size_t)row * D_HEAD + col] = __float2half(0.0f);
                }
            }
            __syncthreads();

            wmma::fill_fragment(S_frag, 0.0f);

            #pragma unroll
            for (int dk = 0; dk < NDK; ++dk) {
                wmma::load_matrix_sync(Af,
                    Q_sh + (size_t)(mq * MMA_M) * D_HEAD + dk * MMA_K, D_HEAD);
                wmma::load_matrix_sync(Bf,
                    K_sh + (size_t)dk * MMA_K, D_HEAD);
                wmma::mma_sync(S_frag, Af, Bf, S_frag);
            }

            for (int e = 0; e < 8; ++e) {
                const int q_row_global = q_base + mq * MMA_M + lane;
                const int k_col_global = k_base + nk * MMA_N + e;

                if (q_row_global >= M || k_col_global >= M) continue;
                if (k_col_global > q_row_global) continue;

                score[(size_t)b * s_S_b + (size_t)q_row_global * s_S_m
                      + (size_t)k_col_global * s_S_n + (size_t)qh * s_S_h]
                    = S_frag.x[e] * scale;
            }
            __syncthreads();
        }
    }
}

// Multiple CTAs per head: saturate all SMs on V100
// For B=1, H=16, we need 80 CTAs → 5 CTAs per head
// Grid dimension: [cta_per_head, B * n_q_heads, 1]
template <int Q_TILE, int K_TILE, int D_HEAD, int CTA_PER_HEAD>
__global__ void __launch_bounds__(512) compute_block_score_gemm_kernel_f16_multi_cta(
    const half * __restrict__ mean_Q,
    const half * __restrict__ mean_K,
    float sm_scale,
    float * __restrict__ score,
    int B, int M, int n_q_heads, int n_k_heads,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h)
{
    using namespace nvcuda;

    constexpr int MMA_M = 32, MMA_N = 8, MMA_K = 16;
    constexpr int NDK = D_HEAD / MMA_K;
    constexpr int NNK = K_TILE / MMA_N;

    // blockIdx.x = which CTA within this head (0..CTA_PER_HEAD-1)
    // blockIdx.y = (b * n_q_heads + qh)
    const int cta_id = blockIdx.x;
    const int b = blockIdx.y / n_q_heads;
    const int qh = blockIdx.y % n_q_heads;
    if (b >= B) return;

    const int kh = qh * n_k_heads / n_q_heads;

    const int wid = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int mq = wid / NNK;
    const int nk = wid % NNK;

    extern __shared__ unsigned char smem_raw[];
    half * Q_sh = reinterpret_cast<half*>(smem_raw);
    half * K_sh = Q_sh + (size_t)Q_TILE * D_HEAD;

    wmma::fragment<wmma::accumulator, MMA_M, MMA_N, MMA_K, float> S_frag;
    wmma::fragment<wmma::matrix_a, MMA_M, MMA_N, MMA_K, half, wmma::row_major> Af;
    wmma::fragment<wmma::matrix_b, MMA_M, MMA_N, MMA_K, half, wmma::col_major> Bf;

    const float scale = sm_scale * 1.4426950408889634f;

    const int n_qt = (M + Q_TILE - 1) / Q_TILE;
    // Each CTA processes a strided subset of Q tiles
    for (int qt = cta_id; qt < n_qt; qt += CTA_PER_HEAD) {
        const int q_base = qt * Q_TILE;
        const int q_limit = min(M - q_base, Q_TILE);

        for (int idx = threadIdx.x; idx < Q_TILE * D_HEAD; idx += blockDim.x) {
            const int row = idx / D_HEAD;
            const int col = idx % D_HEAD;
            if (row < q_limit) {
                Q_sh[idx] = mean_Q[
                    (size_t)b * s_mQ_b + (size_t)(q_base + row) * s_mQ_m
                    + (size_t)qh * s_mQ_h + col];
            } else {
                Q_sh[idx] = __float2half(0.0f);
            }
        }
        __syncthreads();

        for (int kt = 0; kt <= qt; ++kt) {
            const int k_base = kt * K_TILE;
            const int k_limit = min(M - k_base, K_TILE);

            for (int idx = threadIdx.x; idx < K_TILE * D_HEAD; idx += blockDim.x) {
                const int row = idx / D_HEAD;
                const int col = idx % D_HEAD;
                if (row < k_limit) {
                    K_sh[(size_t)row * D_HEAD + col] = mean_K[
                        (size_t)b * s_mK_b + (size_t)(k_base + row) * s_mK_m
                        + (size_t)kh * s_mK_h + col];
                } else {
                    K_sh[(size_t)row * D_HEAD + col] = __float2half(0.0f);
                }
            }
            __syncthreads();

            wmma::fill_fragment(S_frag, 0.0f);

            #pragma unroll
            for (int dk = 0; dk < NDK; ++dk) {
                wmma::load_matrix_sync(Af,
                    Q_sh + (size_t)(mq * MMA_M) * D_HEAD + dk * MMA_K, D_HEAD);
                wmma::load_matrix_sync(Bf,
                    K_sh + (size_t)dk * MMA_K, D_HEAD);
                wmma::mma_sync(S_frag, Af, Bf, S_frag);
            }

            for (int e = 0; e < 8; ++e) {
                const int q_row_global = q_base + mq * MMA_M + lane;
                const int k_col_global = k_base + nk * MMA_N + e;

                if (q_row_global >= M || k_col_global >= M) continue;
                if (k_col_global > q_row_global) continue;

                score[(size_t)b * s_S_b + (size_t)q_row_global * s_S_m
                      + (size_t)k_col_global * s_S_n + (size_t)qh * s_S_h]
                    = S_frag.x[e] * scale;
            }
            __syncthreads();
        }
    }
}

extern "C" void launch_compute_block_score_gemm_f16(
    const void * mean_Q, const void * mean_K, float sm_scale,
    void * score,
    int batch, int n_q_heads, int n_k_heads,
    int M, int head_dim,
    int s_mQ_b, int s_mQ_m, int s_mQ_h,
    int s_mK_b, int s_mK_m, int s_mK_h,
    int s_S_b, int s_S_m, int s_S_n, int s_S_h,
    cudaStream_t stream)
{
    constexpr int Q_TILE = 64, K_TILE = 64, D_HEAD = 128;

    if (head_dim != 128) return;

    // Auto-tune: 5 CTAs per head for 80 SMs on V100
    const int n_heads = batch * n_q_heads;
    const int cta_per_head = (n_heads <= 16) ? 5 : 1;

    dim3 grid(cta_per_head, n_heads, 1);
    dim3 block(512);

    size_t smem = sizeof(half) * (Q_TILE * D_HEAD + K_TILE * D_HEAD);

    if (cta_per_head == 5) {
        compute_block_score_gemm_kernel_f16_multi_cta<Q_TILE, K_TILE, D_HEAD, 5>
            <<<grid, block, smem, stream>>>(
            (const half *)mean_Q, (const half *)mean_K, sm_scale,
            (float *)score,
            batch, M, n_q_heads, n_k_heads,
            s_mQ_b, s_mQ_m, s_mQ_h,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h);
    } else {
        compute_block_score_gemm_kernel_f16<Q_TILE, K_TILE, D_HEAD>
            <<<grid, block, smem, stream>>>(
            (const half *)mean_Q, (const half *)mean_K, sm_scale,
            (float *)score,
            batch, M, n_q_heads, n_k_heads,
            s_mQ_b, s_mQ_m, s_mQ_h,
            s_mK_b, s_mK_m, s_mK_h,
            s_S_b, s_S_m, s_S_n, s_S_h);
    }
}

} // namespace flashprefill
} // namespace dflash27b

#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
