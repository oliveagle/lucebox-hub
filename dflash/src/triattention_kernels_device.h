// triattention_kernels_device.h — Device-only code for TriAttention
//
// This file contains only __device__ code that compiles for both CUDA and HIP.
// It does NOT include any CUDA/HIP runtime headers.

#pragma once

#include <stdint.h>

// Device function attributes - use HIPCC for HIP, CUDA-specific for CUDA
#if defined(__HIP_DEVICE_COMPILE__)
#define TRIA_DEVICE __device__
#define TRIA_DEVICE_INLINE __device__ __forceinline__
#else
#define TRIA_DEVICE __device__
#define TRIA_DEVICE_INLINE __device__ __forceinline__
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
TRIA_DEVICE_INLINE float tria_bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
#if defined(__CUDA_ARCH__)
    f = __uint_as_float(bits);
#elif defined(__HIP_DEVICE_COMPILE__)
    f = __uint_as_float(bits);
#else
    union u32 { uint32_t b; float f; } conv = {bits};
    f = conv.f;
#endif
    return f;
}

// Geometric offsets precomputed once per kernel launch.
TRIA_DEVICE __constant__ float tria_offsets_dev[TRIA_N_OFFSETS];