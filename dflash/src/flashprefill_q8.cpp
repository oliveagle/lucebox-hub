// ggml flash_attn_ext-based FlashPrefill implementation.
//
// Provides flash_prefill_forward_q8() — a portable alternative to the custom
// BF16 WMMA flashprefill kernels. Uses ggml's built-in flash_attn_ext. The
// ggml CUDA/HIP FA kernels require F32 Q/output while still supporting half
// K/V, so this path widens Q and the temporary attention output inside the
// graph and copies the result back to the caller's half/F32 output buffer.
//
// V100 Optimization: Uses ggml_flash_attn_sparse with a registered WMMA
// kernel, bypassing ggml's op dispatch overhead. This provides block-sparse
// attention with DFlash's optimized WMMA kernels on Volta.

#include "flashprefill.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace dflash27b {
namespace flashprefill {

namespace {
constexpr int CHUNK_S = 4096;
}

// Data movement optimization: minimize intermediate tensor operations.
//
// Key insights:
// 1. ggml_permute creates a VIEW (strides don't match new shape) - non-contiguous
// 2. ggml_cont IS needed after permute because fattn-sparse conversion kernels
//    (k_f32_to_bf16_transpose_sh, k_f16_to_bf16_transpose_sh) use flat contiguous
//    indexing - they assume data is in [B,H,S,D] row-major layout
// 3. Use ggml_cast for type conversion - simpler than ggml_cpy + new_tensor
// 4. Conditional type conversion - skip when input already matches required type

int flash_prefill_forward_q8(
    ggml_backend_t backend,
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale,
    ggml_type qkv_type,
    const FlashPrefillConfig & cfg)
{
    // Register pFlash sparse kernel once at first call
    // This enables ggml_flash_attn_sparse to route directly to DFlash WMMA kernels
    {
        extern void pflash_register_ggml_kernel();
        static bool registered = false;
        if (!registered) {
            pflash_register_ggml_kernel();
            registered = true;
        }
    }

    const int S  = seq_len;
    const int H  = n_q_heads;
    const int Hk = n_k_heads;
    const int D  = head_dim;
    [[maybe_unused]] const int B  = batch;

    if (qkv_type != GGML_TYPE_F16 && qkv_type != GGML_TYPE_BF16
        && qkv_type != GGML_TYPE_F32) {
        std::fprintf(stderr, "[flashprefill_q8] unsupported qkv_type=%s\n",
                     ggml_type_name(qkv_type));
        return -1;
    }

    // Use alpha from config for block-sparse attention
    // alpha >= 1.0 means "select all blocks" = dense attention equivalent
    // alpha < 1.0 enables block-sparse selection (fewer blocks = faster)
    float alpha = cfg.alpha > 0.0f ? cfg.alpha : 0.12f;

    // Build ggml graph for sparse FA
    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * 32
                  + ggml_graph_overhead_custom(128, false)
                  + 64 * 1024;
    ip.no_alloc = true;
    ggml_context * ctx = ggml_init(ip);
    if (!ctx) {
        std::fprintf(stderr, "[flashprefill_q8] ggml_init failed\n");
        return -1;
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 128, false);

    // External tensors: Q/K/V [D, H, S] contiguous layout
    ggml_tensor * Q_ext = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
    Q_ext->data = const_cast<void *>(Q);
    ggml_set_name(Q_ext, "Q_ext");
    ggml_set_input(Q_ext);

    ggml_tensor * K_ext = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
    K_ext->data = const_cast<void *>(K);
    ggml_set_name(K_ext, "K_ext");
    ggml_set_input(K_ext);

    ggml_tensor * V_ext = ggml_new_tensor_3d(ctx, qkv_type, D, Hk, S);
    V_ext->data = const_cast<void *>(V);
    ggml_set_name(V_ext, "V_ext");
    ggml_set_input(V_ext);

    ggml_tensor * O_ext = ggml_new_tensor_3d(ctx, qkv_type, D, H, S);
    O_ext->data = O;
    ggml_set_name(O_ext, "O_ext");
    ggml_set_output(O_ext);

    // Prepare Q/K/V tensors for flash_attn_sparse
    // flash_attn_sparse expects [D, S, H] layout (permuted from [D, H, S])
    //
    // IMPORTANT: ggml_cont IS needed after permute.
    // fattn-sparse conversion kernels use flat contiguous indexing:
    //   k_f32_to_bf16_transpose_sh uses src_idx = ((b*H + h)*S + s)*D + d
    // This assumes data is in contiguous [B,H,S,D] row-major layout.
    // Without ggml_cont, the permuted tensor has non-matching strides,
    // causing illegal memory access.
    ggml_tensor * Q_perm = ggml_cont(ctx, ggml_permute(ctx, Q_ext, 0, 2, 1, 3));  // [D,H,S] → [D,S,H]
    ggml_tensor * K_perm = ggml_cont(ctx, ggml_permute(ctx, K_ext, 0, 2, 1, 3));
    ggml_tensor * V_perm = ggml_cont(ctx, ggml_permute(ctx, V_ext, 0, 2, 1, 3));

    // Convert types to match flash_attn_sparse requirements: Q → F32, K/V → F16
    // OPTIMIZATION: Use ggml_cast instead of ggml_cpy + new_tensor
    // ggml_cast creates a type conversion operation in a single step
    ggml_tensor * Q_fa = (qkv_type == GGML_TYPE_F32) ? Q_perm :
                         ggml_cast(ctx, Q_perm, GGML_TYPE_F32);

    ggml_tensor * K_fa = (qkv_type == GGML_TYPE_F16) ? K_perm :
                         ggml_cast(ctx, K_perm, GGML_TYPE_F16);

    ggml_tensor * V_fa = (qkv_type == GGML_TYPE_F16) ? V_perm :
                         ggml_cast(ctx, V_perm, GGML_TYPE_F16);

    // Call ggml_flash_attn_sparse — routes to registered pFlash kernel
    // which uses DFlash block-sparse attention with WMMA kernels
    ggml_tensor * attn = ggml_flash_attn_sparse(ctx, Q_fa, K_fa, V_fa, scale, alpha);
    // attn shape: [D, S, H, B], type: F32

    // Output: write to O_ext
    // No output permute needed - attn result [D,S,H,B] already matches ggml convention
    // and the backend will handle the stride correctly
    //
    // OPTIMIZATION: Simplified output path
    // - No O_perm permute (was unnecessary)
    // - Direct copy from F32 attn result to O_ext
    ggml_tensor * final_op;
    if (qkv_type != GGML_TYPE_F32) {
        // Convert F32 → target type
        ggml_tensor * O_conv = ggml_cast(ctx, attn, qkv_type);
        final_op = ggml_cpy(ctx, O_conv, O_ext);
    } else {
        // Direct copy to output buffer
        final_op = ggml_cpy(ctx, attn, O_ext);
    }

    ggml_build_forward_expand(gf, final_op);

    ggml_gallocr_t galloc = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));

    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        std::fprintf(stderr, "[flashprefill_q8] graph alloc failed\n");
        ggml_free(ctx);
        ggml_gallocr_free(galloc);
        return -1;
    }

    ggml_backend_graph_compute(backend, gf);
    ggml_backend_synchronize(backend);

    ggml_free(ctx);
    ggml_gallocr_free(galloc);
    return 0;
}

} // namespace flashprefill
} // namespace dflash27b