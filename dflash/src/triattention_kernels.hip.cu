// triattention_kernels.hip.cu — HIP/ROCm host implementation for TriAttention GPU scoring
//
// Instantiates the scoring kernel from triattention_kernels.h and provides
// the host-side launcher function ggml_hip_tria_score().
//
// Porting changes from CUDA version:
//   - hipStream_t instead of cudaStream_t
//   - hip_* instead of cuda* for all runtime API calls
//   - __shfl_xor, __ballot, etc. (same on HIP)
//   - hip_bfloat16 and rocwmma if needed (same format as __nv_bfloat16)
//   - __uint_as_float works on HIP

#include "triattention_kernels.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash27b {
namespace triattention {

// ── bf16 to f32 conversion kernel ─────────────────────────────────────

template <int THREADS = 256>
__global__ void tria_bf16_convert_kernel(
    const uint16_t * __restrict__ src,   /* [tensor_head_dim, seq_len, n_head_kv] */
    float          * dst_real,           /* [n_head_kv, seq_len, fc] */
    float          * dst_imag,           /* [n_head_kv, seq_len, fc] */
    int             n_head_kv,
    int             seq_len,
    int             tensor_head_dim,
    int             fc,
    int             kv_start)
{
    const int tid  = threadIdx.x;
    const int head = blockIdx.x;
    const int pos  = blockIdx.y;

    if (head >= n_head_kv || pos >= seq_len) return;

    const size_t src_stride_h = (size_t)seq_len * tensor_head_dim;
    const size_t src_stride_s = tensor_head_dim;
    const size_t src_base = (size_t)head * src_stride_h + (size_t)(kv_start + pos) * src_stride_s;

    for (int f = tid; f < fc; f += THREADS) {
        const size_t dst_idx = ((size_t)head * seq_len + pos) * fc + f;
        dst_real[dst_idx] = tria_bf16_to_f32(src[src_base + f]);
        dst_imag[dst_idx] = tria_bf16_to_f32(src[src_base + fc + f]);
    }
}

// ── Geometric offsets constant ────────────────────────────────────────

__constant__ float tria_offsets_const[TRIA_N_OFFSETS];

// ── Scoring kernel (fixed FC template) ────────────────────────────────

template <int FC>
__global__ void tria_score_kernel_impl(
    const float * k_real,
    const float * k_imag,
    const float * q_stats,      /* [n_full_attn][FC][3] */
    const float * omega,        /* [FC] */
    const int   * key_pos,
    float       * offsets,
    int           cur_pos,
    int           seq_len,
    int           n_kv_heads,
    int           n_full_attn,
    float       * scores_out)   /* [n_full_attn][n_kv_heads][seq_len] */
{
    const int zh = blockIdx.x;
    const int tid = threadIdx.x;

    if (zh >= n_kv_heads * seq_len) return;

    const int kv_head = zh / seq_len;
    const int s = zh % seq_len;

    // Shared memory for qma.
    __shared__ float s_qma[FC];

    // Process each layer.
    for (int layer = 0; layer < n_full_attn; layer++) {
        const float * q_mean_real = q_stats + ((size_t)layer * FC * 3);
        const float * q_mean_imag = q_stats + ((size_t)layer * FC * 3) + FC;
        const float * q_abs_mean = q_stats + ((size_t)layer * FC * 3) + 2 * FC;

        // Thread 0: compute qma.
        if (tid == 0) {
            #pragma unroll
            for (int f = 0; f < FC; f++) {
                float qr = q_mean_real[f];
                float qi = q_mean_imag[f];
                s_qma[f] = sqrtf(qr * qr + qi * qi);
            }
        }
        __syncthreads();

        float base_delta = (float)(cur_pos - key_pos[s]);

        float local_trig = 0.0f;
        float local_extra = 0.0f;

        const float * kr = k_real + ((size_t)kv_head * seq_len + s) * FC;
        const float * ki = k_imag + ((size_t)kv_head * seq_len + s) * FC;

        for (int f = tid; f < FC; f += blockDim.x) {
            float ka_f = sqrtf(kr[f] * kr[f] + ki[f] * ki[f]);

            float rel_real = q_mean_real[f] * kr[f] + q_mean_imag[f] * ki[f];
            float rel_imag = q_mean_imag[f] * kr[f] - q_mean_real[f] * ki[f];
            float phi = atan2f(rel_imag, rel_real);

            float amp = s_qma[f] * ka_f;

            float residual = q_abs_mean[f] - s_qma[f];
            if (residual < 0.0f) residual = 0.0f;
            local_extra += residual * ka_f;

            float trig_contrib = 0.0f;
            #pragma unroll
            for (int o = 0; o < TRIA_N_OFFSETS; o++) {
                float delta = base_delta + offsets[o];
                trig_contrib += amp * cosf(delta * omega[f] + phi);
            }
            local_trig += trig_contrib;
        }

        // Warp reduction (same on HIP as CUDA).
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_trig  += __shfl_xor(local_trig,  mask);
            local_extra += __shfl_xor(local_extra, mask);
        }

        if (tid == 0) {
            float score = local_trig / (float)TRIA_N_OFFSETS + local_extra;
            scores_out[((size_t)layer * n_kv_heads + kv_head) * seq_len + s] = score;
        }

        __syncthreads();
    }
}

// ── Max-pool kernel ───────────────────────────────────────────────────

template <int THREADS>
__global__ void tria_max_pool_kernel_impl(
    const float * scores,   /* [n_full_attn, n_kv_heads, seq_len] */
    float       * pooled,   /* [n_full_attn, seq_len] */
    int           n_full_attn,
    int           n_kv_heads,
    int           seq_len)
{
    const int layer = blockIdx.x;
    const int pos   = blockIdx.y;
    const int tid   = threadIdx.x;

    if (layer >= n_full_attn || pos >= seq_len) return;

    float local_max = -1e30f;
    for (int h = tid; h < n_kv_heads; h += THREADS) {
        const int idx = ((size_t)layer * n_kv_heads + h) * seq_len + pos;
        local_max = fmaxf(local_max, scores[idx]);
    }

    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, mask));
    }

    __shared__ float s_max[32];
    if (tid < 32) s_max[tid] = local_max;
    __syncthreads();

    if (tid == 0) {
        float final_max = -1e30f;
        for (int i = 0; i < min(n_kv_heads, 32); i++) {
            final_max = fmaxf(final_max, s_max[i]);
        }
        pooled[layer * seq_len + pos] = final_max;
    }
}

// ── Layer average kernel ───────────────────────────────────────────────

template <int THREADS>
__global__ void tria_layer_avg_kernel_impl(
    const float * pooled,   /* [n_full_attn, seq_len] */
    float       * combined, /* [seq_len] */
    int           n_full_attn,
    int           seq_len)
{
    const int pos = blockIdx.x * THREADS + threadIdx.x;
    if (pos >= seq_len) return;

    float sum = 0.0f;
    for (int l = 0; l < n_full_attn; l++) {
        sum += pooled[l * seq_len + pos];
    }
    combined[pos] = sum / (float)n_full_attn;
}

// ── HIPRTC compilation helpers (optional) ─────────────────────────────

// For HIP builds, we provide the same API as CUDA.
// The functions are in the same namespace for ABI compatibility.

// ── Main entry point ──────────────────────────────────────────────────

bool ggml_hip_tria_score(
    const void * k_bf16_gpu,
    const float * d_q_stats,
    const float * d_omega,
    const int   * d_key_pos,
    int           cur_pos,
    int           n_full_attn,
    int           n_kv_heads,
    int           tensor_head_dim,
    int           fc,
    int           seq_len,
    int           kv_start,
    float       * d_scores_out,
    int           gqa,
    hipStream_t   stream)
{
    (void)gqa;
    (void)kv_start;
    (void)d_scores_out;

    if (!k_bf16_gpu || !d_q_stats || !d_omega || !d_key_pos) {
        std::fprintf(stderr, "[TriAttention HIP] null pointer passed\n");
        return false;
    }
    if (fc > TRIA_MAX_FC || fc <= 0) {
        std::fprintf(stderr, "[TriAttention HIP] fc=%d out of range [1, %d]\n",
                     fc, TRIA_MAX_FC);
        return false;
    }
    if (n_kv_heads <= 0 || seq_len <= 0 || n_full_attn <= 0) {
        std::fprintf(stderr, "[TriAttention HIP] invalid dimensions\n");
        return false;
    }

    std::fprintf(stderr, "[TriAttention HIP] Starting GPU scoring: fc=%d seq_len=%d "
                "n_kv_heads=%d n_full_attn=%d\n",
                fc, seq_len, n_kv_heads, n_full_attn);

    const size_t k_nelems = (size_t)n_kv_heads * seq_len * fc;

    float * d_k_real = nullptr;
    float * d_k_imag = nullptr;

    hipError_t err = hipMalloc(&d_k_real, k_nelems * sizeof(float));
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMalloc k_real failed: %s\n",
                     hipGetErrorString(err));
        return false;
    }
    err = hipMalloc(&d_k_imag, k_nelems * sizeof(float));
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMalloc k_imag failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_k_real);
        return false;
    }

    // Launch bf16 conversion kernel.
    dim3 conv_grid(n_kv_heads, seq_len, 1);
    hipLaunchKernelGGL(tria_bf16_convert_kernel<256>,
        conv_grid, 256, 0, stream,
        static_cast<const uint16_t*>(k_bf16_gpu),
        d_k_real, d_k_imag,
        n_kv_heads, seq_len, tensor_head_dim, fc, kv_start);

    err = hipPeekAtLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] bf16 convert kernel failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    std::fprintf(stderr, "[TriAttention HIP] bf16 conversion done\n");

    // Allocate scoring buffers.
    float * d_scores = nullptr;
    err = hipMalloc(&d_scores, (size_t)n_full_attn * n_kv_heads * seq_len * sizeof(float));
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMalloc scores failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    float * d_offsets = nullptr;
    err = hipMalloc(&d_offsets, TRIA_N_OFFSETS * sizeof(float));
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMalloc offsets failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    float h_offsets[TRIA_N_OFFSETS];
    for (int i = 0; i < TRIA_N_OFFSETS; i++) {
        h_offsets[i] = (float)(1 << i);
    }
    err = hipMemcpyAsync(d_offsets, h_offsets, TRIA_N_OFFSETS * sizeof(float),
                         hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMemcpy offsets failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_offsets);
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    // Launch scoring kernel.
    const int n_blocks = n_kv_heads * seq_len;
    const int n_threads = 64;

    if (fc == 32) {
        hipLaunchKernelGGL(tria_score_kernel_impl<32>,
            dim3(n_blocks), dim3(n_threads), 0, stream,
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 64) {
        hipLaunchKernelGGL(tria_score_kernel_impl<64>,
            dim3(n_blocks), dim3(n_threads), 0, stream,
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 16) {
        hipLaunchKernelGGL(tria_score_kernel_impl<16>,
            dim3(n_blocks), dim3(n_threads), 0, stream,
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 128) {
        hipLaunchKernelGGL(tria_score_kernel_impl<128>,
            dim3(n_blocks), dim3(n_threads), 0, stream,
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else {
        // Generic fallback.
        hipLaunchKernelGGL(tria_score_kernel_impl<64>,
            dim3(n_blocks), dim3(n_threads), 0, stream,
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    }

    err = hipPeekAtLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] scoring kernel failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_offsets);
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    std::fprintf(stderr, "[TriAttention HIP] scoring kernel done\n");

    // Max-pool across KV heads.
    float * d_pooled = nullptr;
    err = hipMalloc(&d_pooled, (size_t)n_full_attn * seq_len * sizeof(float));
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMalloc pooled failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_offsets);
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    dim3 pool_grid(n_full_attn, seq_len, 1);
    hipLaunchKernelGGL(tria_max_pool_kernel_impl<64>,
        pool_grid, dim3(64), 0, stream,
        d_scores, d_pooled, n_full_attn, n_kv_heads, seq_len);

    err = hipPeekAtLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] max-pool kernel failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_pooled);
        hipFree(d_offsets);
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    // Average across layers.
    const int avg_blocks = (seq_len + 63) / 64;
    hipLaunchKernelGGL(tria_layer_avg_kernel_impl<64>,
        dim3(avg_blocks), dim3(64), 0, stream,
        d_pooled, d_scores, n_full_attn, seq_len);

    err = hipPeekAtLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] layer avg kernel failed: %s\n",
                     hipGetErrorString(err));
        hipFree(d_pooled);
        hipFree(d_offsets);
        hipFree(d_scores);
        hipFree(d_k_real);
        hipFree(d_k_imag);
        return false;
    }

    // Copy to output.
    if (d_scores_out != d_scores) {
        size_t combined_bytes = (size_t)seq_len * sizeof(float);
        err = hipMemcpyAsync(d_scores_out, d_scores, combined_bytes,
                             hipMemcpyDeviceToDevice, stream);
        if (err != hipSuccess) {
            std::fprintf(stderr, "[TriAttention HIP] hipMemcpy combined scores failed: %s\n",
                         hipGetErrorString(err));
            hipFree(d_pooled);
            hipFree(d_offsets);
            hipFree(d_scores);
            hipFree(d_k_real);
            hipFree(d_k_imag);
            return false;
        }
    }

    hipFree(d_pooled);
    hipFree(d_offsets);
    hipFree(d_scores);
    hipFree(d_k_real);
    hipFree(d_k_imag);

    std::fprintf(stderr, "[TriAttention HIP] GPU scoring complete\n");
    return true;
}

// ── Top-K selection on HIP ──────────────────────────────────────────
//
// Uses a simple CPU-based top-K after copying scores to host.
// For production, use hip thrust or a custom sort kernel.

bool ggml_hip_tria_topk(
    const float * d_scores,
    int           seq_len,
    int           k,
    int         * d_topk_out,
    int         * d_count_out,
    hipStream_t   stream)
{
    if (!d_scores || !d_topk_out || k <= 0) {
        std::fprintf(stderr, "[TriAttention HIP] tria_topk: invalid args\n");
        return false;
    }

    k = std::min(k, seq_len);

    // Copy scores to host.
    std::vector<float> h_scores(seq_len);
    hipError_t err = hipMemcpy(h_scores.data(), d_scores, seq_len * sizeof(float),
                               hipMemcpyDeviceToHost);
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMemcpy scores to host failed: %s\n",
                     hipGetErrorString(err));
        return false;
    }

    // Sort on host and select top-K.
    std::vector<std::pair<float, int>> indexed(seq_len);
    for (int i = 0; i < seq_len; i++) {
        indexed[i] = {h_scores[i], i};
    }
    std::partial_sort(indexed.begin(), indexed.begin() + k, indexed.end(),
                       [](const auto & a, const auto & b) { return a.first > b.first; });

    std::vector<int> h_topk(k);
    for (int i = 0; i < k; i++) {
        h_topk[i] = indexed[i].second;
    }

    // Copy back to device.
    err = hipMemcpy(d_topk_out, h_topk.data(), k * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] hipMemcpy topk to device failed: %s\n",
                     hipGetErrorString(err));
        return false;
    }

    if (d_count_out) {
        int h_count = k;
        hipMemcpy(d_count_out, &h_count, sizeof(int), hipMemcpyHostToDevice);
    }

    std::fprintf(stderr, "[TriAttention HIP] Top-K done: k=%d\n", k);
    return true;
}

// ── GPU KV compaction ───────────────────────────────────────────────

__global__ void tria_compact_kv_kernel_impl(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes)
{
    const int dst_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_pos >= actual_keep) return;

    const int src_pos = keep_indices[dst_pos];
    const size_t head_stride = (size_t)max_ctx * head_bytes;
    const size_t src_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + src_pos) * head_bytes;
    const size_t dst_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + dst_pos) * head_bytes;

    char * dst = static_cast<char*>(cache_data) + dst_offset;
    const char * src = static_cast<const char*>(cache_data) + src_offset;
    // Copy element by element (simplified).
    for (size_t b = 0; b < head_bytes; b++) {
        dst[b] = src[b];
    }
}

bool ggml_hip_tria_compact_kv(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len,
    hipStream_t   stream)
{
    const int n_blocks = (actual_keep + 255) / 256;
    hipLaunchKernelGGL(tria_compact_kv_kernel_impl,
        dim3(n_blocks), dim3(256), 0, stream,
        cache_data, keep_indices, actual_keep,
        kv_head, max_ctx, kv_start, head_bytes);

    hipError_t err = hipPeekAtLastError();
    if (err != hipSuccess) {
        std::fprintf(stderr, "[TriAttention HIP] compact_kv kernel failed: %s\n",
                     hipGetErrorString(err));
        return false;
    }
    return true;
}

bool ggml_hip_tria_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start,
    hipStream_t   stream)
{
    // Not implemented. Uses CPU-based compaction for now.
    (void)tria_data;
    (void)keep_indices;
    (void)actual_keep;
    (void)tensor_head_dim;
    (void)max_ctx;
    (void)n_kv_heads;
    (void)kv_start;
    (void)stream;
    std::fprintf(stderr, "[TriAttention HIP] tria_compact_tria_bf16: not yet implemented\n");
    return false;
}

}  // namespace triattention
}  // namespace dflash27b