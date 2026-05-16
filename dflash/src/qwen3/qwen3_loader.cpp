// GGUF loader for Qwen3-0.6B drafter. Reads weights from a GGUF file produced
// by `convert_hf_to_gguf.py Qwen/Qwen3-0.6B`. Sets up ggml tensors on the
// requested backend. On GPUs with native BF16 tensor cores (sm_80+), weights
// are kept as BF16; on older GPUs (sm_70 V100, sm_75 Turing) they are converted
// to F16 at load time for WMMA acceleration.
//
// Tensor layout (verified via gguf reader):
//   token_embd.weight                 BF16 [hidden=1024, vocab=151936]
//   output_norm.weight                F32  [hidden]
//   output.weight                     BF16 [hidden, vocab] (lm_head)
//
//   blk.<i>.attn_norm.weight          F32  [hidden]
//   blk.<i>.attn_q.weight             BF16 [hidden, q_dim=2048]
//   blk.<i>.attn_k.weight             BF16 [hidden, kv_dim=1024]
//   blk.<i>.attn_v.weight             BF16 [hidden, kv_dim]
//   blk.<i>.attn_output.weight        BF16 [q_dim, hidden]
//   blk.<i>.attn_q_norm.weight        F32  [head_dim=128]
//   blk.<i>.attn_k_norm.weight        F32  [head_dim]
//   blk.<i>.ffn_norm.weight           F32  [hidden]
//   blk.<i>.ffn_gate.weight           BF16 [hidden, ffn=3072]
//   blk.<i>.ffn_up.weight             BF16 [hidden, ffn]
//   blk.<i>.ffn_down.weight           BF16 [ffn, hidden]
//
// We mmap the GGUF file and copy each tensor's bytes to the backend buffer
// (mirrors the dflash gguf_target_loader pattern).

#include "qwen3_drafter_model.h"
#include "internal.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#if defined(_WIN32)
#if !defined(NOMINMAX)
#define NOMINMAX
#endif
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#if defined(DFLASH27B_BACKEND_CUDA)
#include <cuda_runtime.h>
#endif

namespace dflash27b {

namespace {

// Detect whether the GPU supports native BF16 tensor cores.
// sm_80+ (Ampere+) has native BF16 WMMA; sm_70 (Volta) and sm_75 (Turing) do not.
static bool gpu_has_native_bf16() {
#if defined(DFLASH27B_BACKEND_CUDA)
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        const int sm = prop.major * 10 + prop.minor;
        return sm >= 80;
    }
#endif
    // Fallback: assume no native BF16.
    return false;
}

// Convert an array of bf16 values to fp16 via f32 intermediate.
static void bf16_to_f16_array(const uint16_t * src, uint16_t * dst, size_t n) {
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = ((uint32_t)src[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        uint32_t u;
        std::memcpy(&u, &f, 4);
        uint32_t sign = (u >> 16) & 0x8000;
        int32_t  exp  = ((u >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (u >> 13) & 0x03FF;
        if (exp <= 0)       dst[i] = (uint16_t)sign;
        else if (exp >= 31) dst[i] = (uint16_t)(sign | 0x7C00);
        else                dst[i] = (uint16_t)(sign | (exp << 10) | mant);
    }
}

bool copy_tensor_from_file(gguf_context * gctx, const char * name,
                           const void * mmap_base, size_t data_offset,
                           ggml_tensor * dst,
                           const bool is_bf16_in_file, const bool convert_to_f16) {
    int idx = gguf_find_tensor(gctx, name);
    if (idx < 0) {
        std::fprintf(stderr, "[qwen3-0.6b] missing tensor: %s\n", name);
        return false;
    }
    const size_t off = gguf_get_tensor_offset(gctx, idx);
    const size_t bytes = ggml_nbytes(dst);
    const uint8_t * src = (const uint8_t *)mmap_base + data_offset + off;

    if (is_bf16_in_file && convert_to_f16) {
        // File has BF16 data but we allocated an F16 tensor: convert.
        const size_t n = bytes / 2;  // half the bytes for F16
        std::vector<uint16_t> f16(n);
        bf16_to_f16_array((const uint16_t *)src, f16.data(), n);
        ggml_backend_tensor_set(dst, f16.data(), 0, n * sizeof(uint16_t));
    } else {
        ggml_backend_tensor_set(dst, src, 0, bytes);
    }
    return true;
}

uint32_t get_u32(gguf_context * g, const char * key, uint32_t def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_u32(g, k);
}

float get_f32(gguf_context * g, const char * key, float def) {
    int k = gguf_find_key(g, key);
    if (k < 0) return def;
    return gguf_get_val_f32(g, k);
}

} // namespace

bool load_qwen3_drafter_model(const std::string & path,
                              ggml_backend_t backend,
                              Qwen3DrafterWeights & out) {
    out.backend = backend;

    gguf_init_params iparams{ /*no_alloc=*/ false, /*ctx=*/ nullptr };
    gguf_context * gctx = gguf_init_from_file(path.c_str(), iparams);
    if (!gctx) {
        set_last_error("gguf_init_from_file failed: " + path);
        return false;
    }

    out.n_embd     = (int)get_u32(gctx, "qwen3.embedding_length", 1024);
    out.n_ff       = (int)get_u32(gctx, "qwen3.feed_forward_length", 3072);
    out.n_head     = (int)get_u32(gctx, "qwen3.attention.head_count", 16);
    out.n_head_kv  = (int)get_u32(gctx, "qwen3.attention.head_count_kv", 8);
    out.n_layer    = (int)get_u32(gctx, "qwen3.block_count", 28);
    out.n_ctx_max  = (int)get_u32(gctx, "qwen3.context_length", 40960);
    out.head_dim   = (int)get_u32(gctx, "qwen3.attention.key_length", 128);
    out.rope_theta = get_f32(gctx, "qwen3.rope.freq_base", 1000000.0f);

    // Detect the actual weight tensor type from the GGUF file.
    // Skip the first few tensors which may be metadata (e.g., token_embd.weight is the first weight).
    // We look for token_embd.weight specifically.
    const int embd_idx = gguf_find_tensor(gctx, "token_embd.weight");
    const ggml_type file_tensor_type = (embd_idx >= 0)
        ? gguf_get_tensor_type(gctx, embd_idx)
        : GGML_TYPE_BF16;  // default guess

    const bool file_has_bf16 = (file_tensor_type == GGML_TYPE_BF16);
    const bool file_has_f16 = (file_tensor_type == GGML_TYPE_F16);
    const bool file_has_quantized = !file_has_bf16 && !file_has_f16;

    if (file_has_quantized) {
        const char* type_name = ggml_type_name(file_tensor_type);
        std::fprintf(stderr, "[qwen3-0.6b] ERROR: GGUF file has quantized weights (%s), "
                        "but the drafter only supports BF16/F16 weights.\n", type_name);
        std::fprintf(stderr, "[qwen3-0.6b] Please use a BF16 or F16 GGUF file, "
                        "or convert from HF: python convert_hf_to_gguf.py Qwen/Qwen3-0.6B --outtype f16\n");
        gguf_free(gctx);
        return false;
    }

    // On sm_70/75, convert BF16 weights to F16 for WMMA acceleration.
    const bool gpu_native_bf16 = gpu_has_native_bf16();
    const bool use_f16 = !gpu_native_bf16 && file_has_bf16;

    const ggml_type wtype = use_f16 ? GGML_TYPE_F16 : file_tensor_type;

    std::fprintf(stderr, "[qwen3-0.6b] file_type=%s target_type=%s (native_bf16=%d)\n",
                ggml_type_name(file_tensor_type), ggml_type_name(wtype), gpu_native_bf16);

    // Compute total tensor metadata size for context allocation.
    const int n_layer = out.n_layer;
    const int n_tensors_per_layer = 11;
    const int n_top_tensors = 3;
    const int total_tensors = n_top_tensors + n_layer * n_tensors_per_layer;

    ggml_init_params ip{};
    ip.mem_size = ggml_tensor_overhead() * total_tensors + 16 * 1024;
    ip.mem_buffer = nullptr;
    ip.no_alloc = true;
    out.ctx = ggml_init(ip);

    const int n_embd = out.n_embd;
    const int n_ff   = out.n_ff;
    const int n_head = out.n_head;
    const int n_head_kv = out.n_head_kv;
    const int head_dim  = out.head_dim;
    const int n_vocab   = out.n_vocab;
    const int q_dim     = n_head * head_dim;
    const int kv_dim    = n_head_kv * head_dim;

    // Top-level tensors.
    out.tok_embd = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_vocab);
    out.out_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
    out.output   = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_vocab);
    ggml_set_name(out.tok_embd, "token_embd.weight");
    ggml_set_name(out.out_norm, "output_norm.weight");
    ggml_set_name(out.output,   "output.weight");

    out.layers.resize(n_layer);
    for (int il = 0; il < n_layer; ++il) {
        auto & L = out.layers[il];
        L.attn_norm = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.wq        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, q_dim);
        L.wk        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, kv_dim);
        L.wv        = ggml_new_tensor_2d(out.ctx, wtype, n_embd, kv_dim);
        L.wo        = ggml_new_tensor_2d(out.ctx, wtype, q_dim, n_embd);
        L.q_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.k_norm    = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, head_dim);
        L.ffn_norm  = ggml_new_tensor_1d(out.ctx, GGML_TYPE_F32, n_embd);
        L.ffn_gate  = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_ff);
        L.ffn_up    = ggml_new_tensor_2d(out.ctx, wtype, n_embd, n_ff);
        L.ffn_down  = ggml_new_tensor_2d(out.ctx, wtype, n_ff, n_embd);
    }

    out.buf = ggml_backend_alloc_ctx_tensors(out.ctx, backend);
    if (!out.buf) {
        set_last_error("ggml_backend_alloc_ctx_tensors failed for Qwen3-0.6B drafter");
        gguf_free(gctx);
        ggml_free(out.ctx);
        out.ctx = nullptr;
        return false;
    }

    // mmap the GGUF data section.
    const size_t data_off = gguf_get_data_offset(gctx);
#if defined(_WIN32)
    std::wstring wpath;
    {
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        if (wlen <= 0) {
            set_last_error("MultiByteToWideChar failed for " + path);
            gguf_free(gctx);
            return false;
        }
        wpath.resize(wlen - 1);
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
    }
    HANDLE hFile = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) {
        set_last_error("CreateFileW failed for " + path);
        gguf_free(gctx);
        return false;
    }
    HANDLE hMapping = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(hFile);
    if (!hMapping) {
        set_last_error("CreateFileMappingA failed for " + path);
        gguf_free(gctx);
        return false;
    }
    void * mm = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!mm) {
        set_last_error("MapViewOfFile failed for " + path);
        gguf_free(gctx);
        return false;
    }
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    struct stat st; ::fstat(fd, &st);
    void * mm = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (mm == MAP_FAILED) {
        set_last_error("mmap failed for " + path);
        gguf_free(gctx);
        return false;
    }
#endif

    bool ok = true;
    // All weight tensors in the GGUF file are BF16/F16 (not quantized).
    const bool is_bf16_in_file = file_has_bf16;
    const bool convert_to_f16 = use_f16;
    ok &= copy_tensor_from_file(gctx, "token_embd.weight", mm, data_off, out.tok_embd,
                                is_bf16_in_file, convert_to_f16);
    ok &= copy_tensor_from_file(gctx, "output_norm.weight", mm, data_off, out.out_norm,
                                /*is_bf16_in_file=*/false, convert_to_f16);
    // Qwen3-0.6B ties lm_head to embed; output.weight is optional.
    if (gguf_find_tensor(gctx, "output.weight") >= 0) {
        ok &= copy_tensor_from_file(gctx, "output.weight", mm, data_off, out.output,
                                    is_bf16_in_file, convert_to_f16);
    } else {
        // Tied weights: copy tok_embd data into output tensor
        std::vector<uint8_t> tmp(ggml_nbytes(out.tok_embd));
        ggml_backend_tensor_get(out.tok_embd, tmp.data(), 0, tmp.size());
        ggml_backend_tensor_set(out.output, tmp.data(), 0, tmp.size());
    }
    char nm[128];
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = out.layers[il];
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_norm.weight",   il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.attn_norm,
                                    /*is_bf16_in_file=*/false, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight",      il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wq,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight",      il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wk,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight",      il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wv,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.wo,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_q_norm.weight", il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.q_norm,
                                    /*is_bf16_in_file=*/false, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.attn_k_norm.weight", il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.k_norm,
                                    /*is_bf16_in_file=*/false, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_norm.weight",    il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_norm,
                                    /*is_bf16_in_file=*/false, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight",    il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_gate,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight",      il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_up,
                                    is_bf16_in_file, convert_to_f16);
        std::snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight",    il);
        ok &= copy_tensor_from_file(gctx, nm, mm, data_off, L.ffn_down,
                                    is_bf16_in_file, convert_to_f16);
    }
#if defined(_WIN32)
    UnmapViewOfFile(mm);
#else
    ::munmap(mm, st.st_size);
#endif
    gguf_free(gctx);

    if (!ok) {
        set_last_error("one or more Qwen3-0.6B tensors failed to load");
        ggml_backend_buffer_free(out.buf);
        ggml_free(out.ctx);
        out.buf = nullptr;
        out.ctx = nullptr;
        return false;
    }
    return true;
}

void free_qwen3_drafter_model(Qwen3DrafterWeights & w) {
    if (w.buf) { ggml_backend_buffer_free(w.buf); w.buf = nullptr; }
    if (w.ctx) { ggml_free(w.ctx); w.ctx = nullptr; }
    w.layers.clear();
    w.tok_embd = w.out_norm = w.output = nullptr;
    w.backend = nullptr;
}

} // namespace dflash27b
