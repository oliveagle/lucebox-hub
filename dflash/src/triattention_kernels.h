// triattention_kernels.h — Shared GPU scoring kernel for TriAttention
//
// This header defines the __device__ scoring function used by both CUDA and
// HIP translation units. It is included by the per-backend *.cu files and
// NOT compiled directly.

#pragma once

#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIP__)
#if defined(__HIP__)
// HIP headers provide __shfl_xor, __syncthreads, etc.
#include <hip/hip_runtime.h>
#else
// CUDA headers for device code
#include <cuda_runtime.h>
#endif
#endif

// ── Algorithm constants ────────────────────────────────────────────────

#ifndef TRIA_N_OFFSETS
#define TRIA_N_OFFSETS 17  // geometric: 1, 2, 4, ..., 65536
#endif

#ifndef TRIA_MAX_FC
#define TRIA_MAX_FC 64  // max frequency count (head_dim / 2)
#endif

// ── Device-only helpers ────────────────────────────────────────────────

// Load a bf16 element from a uint16_t pointer, convert to float.
// Works identically on CUDA and HIP (both use IEEE-754 bf16).
__device__ __forceinline__ float tria_bf16_to_f32(uint16_t h) {
    // bfloat16 has the same format as the upper 16 bits of IEEE-754 float32.
    // We shift the bf16 into the upper half and rely on the compiler to
    // interpret the resulting 32-bit bit-pattern as a float.
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    // Reinterpret bits → float without any arithmetic.
    // Both CUDA (as_float) and HIP (__uint_as_float) provide this.
#if defined(__CUDA_ARCH__)
    f = __uint_as_float(bits);
#elif defined(__HIP_DEVICE_COMPILE__)
    f = __uint_as_float(bits);
#else
    // Host fallback (should never be instantiated on host).
    union u32 { uint32_t b; float f; } conv = {bits};
    f = conv.f;
#endif
    return f;
}

// Geometric offsets precomputed once per kernel launch.
__device__ __constant__ float tria_offsets[TRIA_N_OFFSETS];

// ── Core scoring kernel (one thread = one (kv_head, position) pair) ─────
//
// Each thread computes scores for one kv_head and one position s, processing
// all fc frequency bands. The scores across positions are accumulated across
// full-attention layers and eventually normalized.
//
// Parameters (all device pointers):
//   k_real   — [n_kv_heads, seq_len, fc] float32
//   k_imag   — [n_kv_heads, seq_len, fc] float32
//   q_stats  — [n_full_attn, fc, 3] per-layer Q stats (real, imag, abs)
//   omega    — [fc] RoPE frequencies (same for all layers/heads)
//   key_pos  — [seq_len] absolute position of each key
//   cur_pos  — current position (trigger point)
//   fc       — frequency count
//   seq_len  — sequence length
//   n_kv_heads — number of KV heads
//   n_full_attn — number of full-attention layers
//   scores_out   — [n_full_attn, n_kv_heads, seq_len] output scores
//
// Grid: blockIdx.x = kv_head * seq_len + position  (linearized)
//       blockDim.x = 64
// Each block computes scores for ONE (kv_head, position) pair across ALL layers.
__device__ void tria_score_one_position_kernel(
    const float * k_real,
    const float * k_imag,
    const float * q_stats,     /* [n_full_attn][fc][3] */
    const float * omega,
    const int   * key_pos,
    int           cur_pos,
    int           fc,
    int           seq_len,
    int           n_kv_heads,
    int           n_full_attn,
    float       * scores_out   /* [n_full_attn][n_kv_heads][seq_len] */
) {
    // Linearize (kv_head, position) to one global index.
    const int zh = blockIdx.x;
    const int tid = threadIdx.x;

    if (zh >= n_kv_heads * seq_len) return;

    const int kv_head   = zh / seq_len;
    const int s         = zh % seq_len;

    // Compute global K pointer for this head.
    const float * kr = k_real + ((size_t)kv_head * seq_len + s) * fc;
    const float * ki = k_imag + ((size_t)kv_head * seq_len + s) * fc;

    // Thread 0 precomputes qma[fc] and loads base_delta once.
    __shared__ float s_qma[TRIA_MAX_FC];
    __shared__ float s_base_delta;

    if (tid < fc) s_qma[tid] = 0.0f;
    if (tid == 0) s_base_delta = 0.0f;
    __syncthreads();

    // Process each layer. We use a loop over layers since n_full_attn is small.
    for (int layer = 0; layer < n_full_attn; layer++) {
        // Load Q stats for this layer.
        const float * q_mean_real = q_stats + ((size_t)layer * fc * 3);
        const float * q_mean_imag = q_stats + ((size_t)layer * fc * 3) + fc;
        const float * q_abs_mean = q_stats + ((size_t)layer * fc * 3) + 2 * fc;

        // Thread 0: compute qma and base_delta once per layer.
        if (tid < fc && tid == 0) {
            // Compute |E[q_f]| for each frequency band.
            // We only need qma[f] = sqrt(qr^2 + qi^2).
            // qma[f] = q_mean_real[f]*q_mean_real[f] + q_mean_imag[f]*q_mean_imag[f]
            // The real computation uses sqrt, so we need all threads for fc values.
            // For simplicity: compute all fc values in the tid < fc check.
            // Since tid == 0 here, we compute sequentially (fc is at most 64).
            // Better: use warp reduction for sqrt. But fc ≤ 64, sequential is fine.
            // Actually, let's use a loop for all fc values.
            for (int f = 0; f < fc; f++) {
                float qr = q_mean_real[f];
                float qi = q_mean_imag[f];
                s_qma[f] = sqrtf(qr * qr + qi * qi);
            }
            s_base_delta = (float)(cur_pos - key_pos[s]);
        }
        if (tid == 0) {
            s_base_delta = (float)(cur_pos - key_pos[s]);
        }

        // Wait for qma and base_delta to be ready.
        __syncthreads();

        // Each thread computes a subset of fc values for phi, amp, extra.
        // Then we reduce across threads for the final trig sum.
        float local_trig_sum = 0.0f;
        float local_extra   = 0.0f;

        // Process fc in chunks of blockDim.x.
        for (int f = tid; f < fc; f += blockDim.x) {
            float ka_f = sqrtf(kr[f] * kr[f] + ki[f] * ki[f]);

            // Phase: angle(q_mean * conj(k))
            float rel_real = q_mean_real[f] * kr[f] + q_mean_imag[f] * ki[f];
            float rel_imag = q_mean_imag[f] * kr[f] - q_mean_real[f] * ki[f];
            float phi = atan2f(rel_imag, rel_real);

            float amp = s_qma[f] * ka_f;

            // Norm extra: max(0, E[||q||] - ||E[q]||) * ||k||
            float residual = q_abs_mean[f] - s_qma[f];
            if (residual < 0.0f) residual = 0.0f;
            local_extra += residual * ka_f;

            // Trig sum over all geometric offsets.
            float trig_contrib = 0.0f;
            for (int o = 0; o < TRIA_N_OFFSETS; o++) {
                float delta = s_base_delta + tria_offsets[o];
                trig_contrib += amp * cosf(delta * omega[f] + phi);
            }
            local_trig_sum += trig_contrib;
        }

        // Warp reduction for local_trig_sum.
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            local_trig_sum += __shfl_xor(local_trig_sum, mask);
            local_extra   += __shfl_xor(local_extra,   mask);
        }

        // Thread 0 writes the layer score.
        if (tid == 0) {
            float score = local_trig_sum / (float)TRIA_N_OFFSETS + local_extra;
            scores_out[((size_t)layer * n_kv_heads + kv_head) * seq_len + s] = score;
        }

        __syncthreads();
    }
}

// ── Kernel launch wrapper ─────────────────────────────────────────────
//
// Computes scores for all (kv_head, position, layer) triples on GPU.
// This replaces the CPU loop in tria_kv_compress.
//
// Grid: n_kv_heads * seq_len blocks, 64 threads each.
// Writes: [n_full_attn][n_kv_heads][seq_len] scores to GPU.
template <int BLOCK_SIZE = 64>
__global__ void tria_score_kernel(
    const float * k_real,
    const float * k_imag,
    const float * q_stats,
    const float * omega,
    const int   * key_pos,
    int           cur_pos,
    int           fc,
    int           seq_len,
    int           n_kv_heads,
    int           n_full_attn,
    float       * scores_out) {

    tria_score_one_position_kernel(
        k_real, k_imag, q_stats, omega, key_pos,
        cur_pos, fc, seq_len, n_kv_heads, n_full_attn, scores_out);
}