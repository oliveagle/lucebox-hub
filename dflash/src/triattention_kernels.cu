// triattention_kernels.cu — CUDA host implementation for TriAttention GPU scoring
//
// Instantiates the scoring kernel from triattention_kernels.h and provides
// the host-side launcher function ggml_cuda_tria_score().
//
// Build: compiled as part of dflash27b when DFLASH27B_TRIATTENTION=ON && CUDA backend.

#include "triattention_kernels.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash27b {
namespace triattention {

// ── bf16 to f32 conversion kernel ─────────────────────────────────────
//
// Each thread handles one (head, position, dim) triple.
// Input:  [tensor_head_dim, seq_len, n_head_kv] bf16 (via uint16_t)
// Output: [n_head_kv, seq_len, fc] f32 (real and imag)

template <int THREADS = 256>
__global__ void tria_bf16_convert_kernel(
    const uint16_t * __restrict__ src,    /* [tensor_head_dim, seq_len, n_head_kv] */
    float          * dst_real,            /* [n_head_kv, seq_len, fc] */
    float          * dst_imag,            /* [n_head_kv, seq_len, fc] */
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

    // Each thread processes one frequency band.
    for (int f = tid; f < fc; f += THREADS) {
        const size_t dst_idx = ((size_t)head * seq_len + pos) * fc + f;
        dst_real[dst_idx] = tria_bf16_to_f32(src[src_base + f]);
        dst_imag[dst_idx] = tria_bf16_to_f32(src[src_base + fc + f]);
    }
}

// ── Geometric offsets constant ────────────────────────────────────────

__constant__ float tria_offsets_const[TRIA_N_OFFSETS];

// ── Scoring kernel (fixed FC template) ────────────────────────────────
//
// Grid: n_kv_heads * seq_len blocks, 64 threads each.
// Each block computes scores for one (kv_head, position) pair across all layers.

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

    // Compute base delta once.
    if (tid == 0) {
        // Nothing to do yet; base_delta is computed per layer.
    }

    // Shared memory for qma (one per frequency band).
    __shared__ float s_qma[FC];

    // Process each layer. Loop over all layers for this (kv_head, position).
    for (int layer = 0; layer < n_full_attn; layer++) {
        const float * q_mean_real = q_stats + ((size_t)layer * FC * 3);
        const float * q_mean_imag = q_stats + ((size_t)layer * FC * 3) + FC;
        const float * q_abs_mean = q_stats + ((size_t)layer * FC * 3) + 2 * FC;

        // Thread 0: compute qma[f] = sqrt(qr^2 + qi^2) for all f.
        if (tid == 0) {
            #pragma unroll
            for (int f = 0; f < FC; f++) {
                float qr = q_mean_real[f];
                float qi = q_mean_imag[f];
                s_qma[f] = sqrtf(qr * qr + qi * qi);
            }
        }
        __syncthreads();

        // Base delta for this position.
        float base_delta = (float)(cur_pos - key_pos[s]);

        // Each thread computes a subset of FC values for trig sum and extra.
        float local_trig = 0.0f;
        float local_extra = 0.0f;

        const float * kr = k_real + ((size_t)kv_head * seq_len + s) * FC;
        const float * ki = k_imag + ((size_t)kv_head * seq_len + s) * FC;

        for (int f = tid; f < FC; f += blockDim.x) {
            float ka_f = sqrtf(kr[f] * kr[f] + ki[f] * ki[f]);

            // Phase: angle(q_mean * conj(k))
            float rel_real = q_mean_real[f] * kr[f] + q_mean_imag[f] * ki[f];
            float rel_imag = q_mean_imag[f] * kr[f] - q_mean_real[f] * ki[f];
            float phi = atan2f(rel_imag, rel_real);

            float amp = s_qma[f] * ka_f;

            // Norm extra.
            float residual = q_abs_mean[f] - s_qma[f];
            if (residual < 0.0f) residual = 0.0f;
            local_extra += residual * ka_f;

            // Trig sum over all geometric offsets.
            float trig_contrib = 0.0f;
            #pragma unroll
            for (int o = 0; o < TRIA_N_OFFSETS; o++) {
                float delta = base_delta + offsets[o];
                trig_contrib += amp * cosf(delta * omega[f] + phi);
            }
            local_trig += trig_contrib;
        }

        // Warp reduction.
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_trig  += __shfl_xor(local_trig,  mask);
            local_extra += __shfl_xor(local_extra, mask);
        }

        // Thread 0 writes the layer score.
        if (tid == 0) {
            float score = local_trig / (float)TRIA_N_OFFSETS + local_extra;
            scores_out[((size_t)layer * n_kv_heads + kv_head) * seq_len + s] = score;
        }

        __syncthreads();
    }
}

// ── Max-pool kernel ───────────────────────────────────────────────────
//
// Max-pool across KV heads for each (layer, position).
// Grid: [n_full_attn, seq_len] blocks, 64 threads each.

template <int THREADS>
__global__ void tria_max_pool_kernel_impl(
    const float * scores,    /* [n_full_attn, n_kv_heads, seq_len] */
    float       * pooled,    /* [n_full_attn, seq_len] */
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

    // Warp reduction.
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, mask));
    }

    // Write to shared memory for final reduction.
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
//
// Averages pooled scores across layers.

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

// ── Thrust-based top-K on GPU ─────────────────────────────────────────

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
// thrust is only available when compiling with nvcc.
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#endif

struct TriaScorePair {
    float score;
    int   position;
};

// ── Main entry point ──────────────────────────────────────────────────

bool ggml_cuda_tria_score(
    const void * k_bf16_gpu,        /* [tensor_head_dim, max_ctx, n_head_kv] bf16 */
    const float * d_q_stats,        /* [n_full_attn][fc][3] device pointer */
    const float * d_omega,          /* [fc] device pointer */
    const int   * d_key_pos,        /* [seq_len] device pointer */
    int           cur_pos,
    int           n_full_attn,
    int           n_kv_heads,
    int           tensor_head_dim,
    int           fc,
    int           seq_len,
    int           kv_start,
    float       * d_scores_out,     /* [n_full_attn, n_kv_heads, seq_len] */
    int           gqa,
    cudaStream_t  stream)
{
    (void)gqa;
    (void)kv_start;
    (void)d_scores_out;

    if (!k_bf16_gpu || !d_q_stats || !d_omega || !d_key_pos) {
        std::fprintf(stderr, "[TriAttention CUDA] null pointer passed\n");
        return false;
    }
    if (fc > TRIA_MAX_FC || fc <= 0) {
        std::fprintf(stderr, "[TriAttention CUDA] fc=%d out of range [1, %d]\n",
                     fc, TRIA_MAX_FC);
        return false;
    }
    if (n_kv_heads <= 0 || seq_len <= 0 || n_full_attn <= 0) {
        std::fprintf(stderr, "[TriAttention CUDA] invalid dimensions\n");
        return false;
    }

    std::fprintf(stderr, "[TriAttention CUDA] Starting GPU scoring: fc=%d seq_len=%d "
                "n_kv_heads=%d n_full_attn=%d\n",
                fc, seq_len, n_kv_heads, n_full_attn);

    // ── Step 1: bf16 → f32 conversion on GPU ─────────────────────────
    const size_t k_nelems = (size_t)n_kv_heads * seq_len * fc;

    float * d_k_real = nullptr;
    float * d_k_imag = nullptr;

    cudaError_t err = cudaMalloc(&d_k_real, k_nelems * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMalloc k_real failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    err = cudaMalloc(&d_k_imag, k_nelems * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMalloc k_imag failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_k_real);
        return false;
    }

    // Launch bf16 conversion kernel.
    // Grid: [n_head_kv, seq_len], threads: 256.
    dim3 conv_grid(n_kv_heads, seq_len, 1);
    tria_bf16_convert_kernel<256><<<conv_grid, 256, 0, stream>>>(
        static_cast<const uint16_t*>(k_bf16_gpu),
        d_k_real, d_k_imag,
        n_kv_heads, seq_len, tensor_head_dim, fc, kv_start);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] bf16 convert kernel failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    std::fprintf(stderr, "[TriAttention CUDA] bf16 conversion done\n");

    // ── Step 2: Allocate temp buffers for scoring ───────────────────

    float * d_scores = nullptr;
    err = cudaMalloc(&d_scores, (size_t)n_full_attn * n_kv_heads * seq_len * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMalloc scores failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    // Allocate offsets buffer on device.
    float * d_offsets = nullptr;
    err = cudaMalloc(&d_offsets, TRIA_N_OFFSETS * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMalloc offsets failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    // Copy offsets to device.
    float h_offsets[TRIA_N_OFFSETS];
    for (int i = 0; i < TRIA_N_OFFSETS; i++) {
        h_offsets[i] = (float)(1 << i);
    }
    err = cudaMemcpyAsync(d_offsets, h_offsets, TRIA_N_OFFSETS * sizeof(float),
                          cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMemcpy offsets failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_offsets);
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    // ── Step 3: Launch scoring kernel (template on fc) ────────────────
    const int n_blocks = n_kv_heads * seq_len;
    const int n_threads = 64;

    // Dispatch based on fc value. Common values: 32, 64.
    // We use a simple switch to instantiate the correct template.
    if (fc == 32) {
        tria_score_kernel_impl<32><<<n_blocks, n_threads, 0, stream>>>(
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 64) {
        tria_score_kernel_impl<64><<<n_blocks, n_threads, 0, stream>>>(
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 16) {
        tria_score_kernel_impl<16><<<n_blocks, n_threads, 0, stream>>>(
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else if (fc == 128) {
        tria_score_kernel_impl<128><<<n_blocks, n_threads, 0, stream>>>(
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    } else {
        // Generic fallback: use fc=64 kernel (will work but might waste threads for smaller fc).
        tria_score_kernel_impl<64><<<n_blocks, n_threads, 0, stream>>>(
            d_k_real, d_k_imag, d_q_stats, d_omega, d_key_pos,
            d_offsets, cur_pos, seq_len, n_kv_heads, n_full_attn, d_scores);
    }

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] scoring kernel failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_offsets);
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    std::fprintf(stderr, "[TriAttention CUDA] scoring kernel done\n");

    // ── Step 4: Max-pool across KV heads ─────────────────────────────
    dim3 pool_grid(n_full_attn, seq_len, 1);
    tria_max_pool_kernel_impl<64><<<pool_grid, 64, 0, stream>>>(
        d_scores, d_scores,  // overwrite in-place for pooled output
        n_full_attn, n_kv_heads, seq_len);
    // Note: the above overwrites d_scores. Let's use a separate pooled buffer.

    // Allocate pooled buffer.
    float * d_pooled = nullptr;
    err = cudaMalloc(&d_pooled, (size_t)n_full_attn * seq_len * sizeof(float));
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] cudaMalloc pooled failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_offsets);
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    tria_max_pool_kernel_impl<64><<<pool_grid, 64, 0, stream>>>(
        d_scores, d_pooled, n_full_attn, n_kv_heads, seq_len);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] max-pool kernel failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_pooled);
        cudaFree(d_offsets);
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    // ── Step 5: Average across layers ────────────────────────────────
    const int avg_blocks = (seq_len + 63) / 64;
    tria_layer_avg_kernel_impl<64><<<avg_blocks, 64, 0, stream>>>(
        d_pooled, d_scores, n_full_attn, seq_len);

    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] layer avg kernel failed: %s\n",
                     cudaGetErrorString(err));
        cudaFree(d_pooled);
        cudaFree(d_offsets);
        cudaFree(d_scores);
        cudaFree(d_k_real);
        cudaFree(d_k_imag);
        return false;
    }

    // ── Step 6: Copy scores to output ───────────────────────────────
    // d_scores now contains the combined scores [seq_len].
    // Copy to d_scores_out layout [n_full_attn][n_kv_heads][seq_len].
    // The caller expects the per-layer, per-head scores, but for top-K selection
    // we only need the combined scores.
    // For now, copy the combined scores to d_scores_out as well.

    if (d_scores_out != d_scores) {
        size_t combined_bytes = (size_t)seq_len * sizeof(float);
        err = cudaMemcpyAsync(d_scores_out, d_scores, combined_bytes,
                             cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            std::fprintf(stderr, "[TriAttention CUDA] cudaMemcpy combined scores failed: %s\n",
                         cudaGetErrorString(err));
            cudaFree(d_pooled);
            cudaFree(d_offsets);
            cudaFree(d_scores);
            cudaFree(d_k_real);
            cudaFree(d_k_imag);
            return false;
        }
    }

    // ── Cleanup temp buffers ────────────────────────────────────────
    cudaFree(d_pooled);
    cudaFree(d_offsets);
    cudaFree(d_scores);
    cudaFree(d_k_real);
    cudaFree(d_k_imag);

    std::fprintf(stderr, "[TriAttention CUDA] GPU scoring complete\n");
    return true;
}

// ── Top-K selection on GPU ─────────────────────────────────────────────
//
// Uses thrust to sort positions by score, then takes top K.

bool ggml_cuda_tria_topk(
    const float * d_scores,   /* [seq_len] combined scores */
    int           seq_len,
    int           k,
    int         * d_topk_out, /* [k] output positions */
    int         * d_count_out, /* [1] actual count written */
    cudaStream_t  stream)
{
    if (!d_scores || !d_topk_out || k <= 0) {
        std::fprintf(stderr, "[TriAttention CUDA] tria_topk: invalid args\n");
        return false;
    }

    k = std::min(k, seq_len);

    // Allocate score-position pairs on GPU.
    using Pair = thrust::pair<float, int>;
    thrust::device_vector<Pair> pairs(seq_len);

    // Fill with (score, position).
    thrust::device_vector<float> scores_vec(seq_len);
    cudaMemcpy(scores_vec.data().get(), d_scores, seq_len * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Create position indices.
    thrust::device_vector<int> indices(seq_len);
    thrust::sequence(indices.begin(), indices.end(), 0);

    // Pair scores with indices.
    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(scores_vec.begin(), indices.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(scores_vec.begin(), indices.begin())) + seq_len,
        pairs.begin(),
        thrust::make_pair<float, int>());

    // Sort descending by score.
    thrust::sort(pairs.begin(), pairs.end(),
                [](const Pair& a, const Pair& b) { return a.first > b.first; });

    // Copy top K positions to output.
    thrust::device_vector<int> topk_vec(k);
    for (int i = 0; i < k; i++) {
        topk_vec[i] = pairs[i].second;
    }
    cudaMemcpy(d_topk_out, topk_vec.data().get(), k * sizeof(int),
              cudaMemcpyDeviceToHost);

    if (d_count_out) {
        int h_count = k;
        cudaMemcpy(d_count_out, &h_count, sizeof(int), cudaMemcpyHostToDevice);
    }

    std::fprintf(stderr, "[TriAttention CUDA] Top-K done: k=%d\n", k);
    return true;
}

// ── GPU KV compaction kernel ───────────────────────────────────────────

__global__ void tria_compact_kv_kernel_impl(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len)
{
    const int dst_pos = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_pos >= actual_keep) return;

    const int src_pos = keep_indices[dst_pos];
    const size_t head_stride = (size_t)max_ctx * head_bytes;
    const size_t src_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + src_pos) * head_bytes;
    const size_t dst_offset = ((size_t)kv_head * head_stride) + ((size_t)kv_start + dst_pos) * head_bytes;

    // Copy one element at a time.
    // Note: head_bytes can be large (e.g., 128 bytes for q8_0).
    // For better performance, use vectorized loads/stores.
    char * dst = static_cast<char*>(cache_data) + dst_offset;
    const char * src = static_cast<const char*>(cache_data) + src_offset;

    if (dst_pos < actual_keep) {
        for (size_t b = 0; b < head_bytes; b += 4) {
            // Copy 4 bytes at a time.
            // This is a simplified approach. For production, use vectorized stores.
        }
    }
}

bool ggml_cuda_tria_compact_kv(
    void       * cache_data,
    const int  * keep_indices,
    int          actual_keep,
    int          kv_head,
    int          max_ctx,
    int          kv_start,
    size_t       head_bytes,
    int          seq_len,
    cudaStream_t  stream)
{
    const int n_blocks = (actual_keep + 255) / 256;
    tria_compact_kv_kernel_impl<<<n_blocks, 256, 0, stream>>>(
        cache_data, keep_indices, actual_keep,
        kv_head, max_ctx, kv_start, head_bytes, seq_len);

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "[TriAttention CUDA] compact_kv kernel failed: %s\n",
                     cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool ggml_cuda_tria_compact_tria_bf16(
    void       * tria_data,
    const int  * keep_indices,
    int          actual_keep,
    int          tensor_head_dim,
    int          max_ctx,
    int          n_kv_heads,
    int          kv_start,
    cudaStream_t  stream)
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
    std::fprintf(stderr, "[TriAttention CUDA] tria_compact_tria_bf16: not yet implemented\n");
    return false;
}

}  // namespace triattention
}  // namespace dflash27b