# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Pre-RoPE K capture for TriAttention**: In `build_full_attn_block()`, capture K after `rms_norm_mul()` but before `ggml_rope_multi()`. Store in `cache.tria_k_pre_rope` buffer (shape `[head_dim, max_ctx, n_head_kv]` bf16). Use `ggml_permute()` to transpose to `[head_dim, n_tokens, n_head_kv]` before copying to cache slot at offset `nb[1] * kv_start`. Gated by `#if defined(DFLASH27B_TRIATTENTION_ENABLED)` compile-time guard.
- **Qwen3.6-27B TriAttention stats**: Pre-built at `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt` (1.6MB). Covers layers 3-63 (every 4th layer), 48 heads each = 768 heads total. `head_dim=256`, `dtype=bfloat16`.
- **DFlash + TriAttention C++ integration**: Include `triattention.h` directly in C++ wrapper. Use full struct definition (not forward declaration) to access nested fields like `layer_budget_scales`.
- **TriAttention KV compression in decode loop**: Integrated in `spec_decode.cpp::run_target_layer_split_dflash_decode()`. After each commit, check `g_tria_state.should_compress(committed)` and call `tria_kv_compress()` if true. Compression happens outside the compute graph via D2H/H2D copies, top-K selection, and in-place cache compaction.


---

## 2026-05-19 - lucebox-hub-gfx1151-dtri-vve
- **Implemented**: DFlash + TriAttention 纯 C++ 集成 (KV cache compression)
- **Files changed**:
  - `dflash/src/triattention_runner.h`: Added `tria_kv_compress()` function declaration for KV compression
  - `dflash/src/triattention_runner.cpp`: Updated `init_triattention_from_env()` to load stats from C library via `tria_load()`, added `TRIATTN_ENABLED` env var check
  - `dflash/src/triattention_compress.cpp`: Created new file implementing `tria_kv_compress()` - reads pre-RoPE K from GPU, scores positions, compacts KV cache in-place
  - `dflash/src/qwen35/spec_decode.cpp`: Integrated KV compression into decode loop - triggers every `divide_length` tokens via `should_compress()`, updates `cache.cur_pos` after compression
  - `dflash/src/internal.h`: Added `tria_last_compressed_pos` field to `TargetCache` for tracking compression state
  - `dflash/CMakeLists.txt`: Added `src/triattention_compress.cpp` to build sources
- **Build verification**: Successfully compiled `libdflash27b.a` and `test_dflash` with TriAttention integration
- **Environment variables**:
  - `TRIATTN_ENABLED=1`: Enable TriAttention KV compression (requires stats file)
  - `TRIATTN_STATS_PATH`: Path to .bin stats file (default: `dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin`)
  - `TRIATTN_KV_BUDGET`: Max tokens to retain (default: 2048)
  - `TRIATTN_DIVIDE_LENGTH`: Compression interval in tokens (default: 128)
  - `TRIATTN_WINDOW_SIZE`: Recent tokens always preserved (default: 128)
- **Learnings**:
  - HIP/CUDA API compatibility: Used macro abstraction (`gpuGetDevice`, `gpuMemcpy`, etc.) to support both CUDA and ROCm
  - C++17 `std::bit_cast` not available on all compilers, used union-based bf16-to-f32 conversion instead
  - KV cache compaction happens in-place via D2H/H2D copies, avoiding memory reallocation
  - The `TriAttentionState` struct uses `stats_ptr` (C library pointer) and `divide_length` (compression interval), not `stats`/`compress_interval`
  - TriAttention C library (`triattention.h/c`) provides `tria_load()` and `tria_free()` for stats lifecycle management

---

## 2026-05-19 - lucebox-hub-gfx1151-0e5
- **Implemented**: Pre-RoPE K capture in DFlash target graph's `build_full_attn_block()` for TriAttention scoring
- **Files changed**:
  - `dflash/src/qwen35/qwen35_target_graph.cpp`: 
    1. Added `tria_k_pre_rope` allocation (bf16 buffer, shape `[head_dim, max_ctx_alloc, n_head_kv]`) in `create_target_cache_partial()` gated by `#if defined(DFLASH27B_TRIATTENTION_ENABLED)`
    2. Added pre-RoPE K capture in `build_full_attn_block()`: after K normalization, before M-RoPE, copy K to `tria_k_pre_rope` at position slot
    3. Added `tria_k_pre_rope` parameter to `build_full_attn_block()` signature, updated both call sites (`build_single_layer` and `build_qwen35_graph`)
    4. Added `tria_k_pre_rope = nullptr;` cleanup in `free_target_cache()`
- **Build verification**: Successfully compiled `libdflash27b.a`, `test_dflash`, `pflash_daemon`
- **Learnings:**
  - `build_full_attn_block()` is called from two call sites with different signatures (one uses named params like `q_tail_capture, q_tail_start`, the other uses default params)
  - The TriAttention pre-RoPE K buffer is separate from the quantized KV cache (uses bf16 for scoring accuracy)
  - Capture happens inline in the compute graph via `ggml_cpy`, no extra host-side copy needed
  - The `#if defined(DFLASH27B_TRIATTENTION_ENABLED)` compile-time guard ensures zero overhead when TriAttention is not enabled
---

## 2026-05-18 - lucebox-hub-gfx1151-z0k.1
- Verified Qwen3.6-27B TriAttention stats file already exists and is valid
- File: `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt` (1.6MB, > 1MB threshold)
- Contains 768 head entries across layers 3-63 (full_attention layers only)
- Successfully loads via torch.load with metadata including head_dim=256, dtype=bfloat16, rope_style=half
- **Environment constraints discovered:**
  - GPU is Tesla GV100 (CC 7.0), driver 535.288.01 supports CUDA 12.2
  - venv PyTorch 2.11+cu130 incompatible with driver (CUDA 13.0 > driver 12.2)
  - lmdeploy venv PyTorch 2.10+cu128 works with GPU
  - AWQ models require gptqmodel, which is incompatible with PyTorch 2.10 due to Autotuner API changes
  - PyTorch 2.11 compiled without CC 7.0 support in cu128 builds
  - **Conclusion**: Pre-built stats file was necessary because on-demand calibration is blocked by env/driver constraints on this hardware
- **Learnings:**
  - Qwen3.6-27B has 16 full_attention layers out of 64 total (every 4th layer: 3,7,11,...,63)
  - The stats file uses only full_attention layers, consistent with Qwen3.6's hybrid attention+SSM architecture
  - Always check for existing pre-built stats before running calibration
---

## 2026-05-18 - lucebox-hub-gfx1151-xlc
- **Implemented**: DFlash + TriAttention integration Phase 1 (Foundation)
- **Files changed**:
  - `dflash/src/internal.h`: Added TriAttentionState to TargetCache, tria_k_pre_rope placeholder
  - `dflash/src/triattention_runner.h`: Fixed include to use full tria_stats definition (was forward decl)
  - `dflash/src/qwen35/qwen35_daemon.cpp`: Added TriAttention initialization at daemon startup
  - `docs/dflash_triattention_integration_status_20260518.md`: Integration status document
- **Build verification**: Successfully compiled `test_dflash` with TriAttention enabled
  - TriAttention C library linked: `libtriattention.a` (17.4K)
  - Symbols verified: `tria_load`, `tria_free`, `tria_score_kv_head`, `init_triattention_from_env`
- **Stats files verified**: `qwen3.5-27b.bin`, `qwen3-8b.bin`, `qwen3-1.7b.bin` available
- **Learnings:**
  - C++ integration requires including `triattention.h` directly (not forward declaration) to access nested fields like `layer_budget_scales`
  - Full KV scoring/pruning requires architectural changes: capture pre-RoPE K, compact cache in-place, remap positions
  - Recommended approach: Use vLLM + TriAttention (working) + DFlash (working) as separate paths
