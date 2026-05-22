# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-22] - lucebox-hub-gfx1151-0s1
- **Issue**: DFlash+TriAttention (29.05 tok/s) slower than DFlash only (30.49 tok/s) - 5% degradation
- **Root causes identified**:
  1. Compression triggers every 128 tokens regardless of KV budget (wasteful when cache < kv_budget)
  2. GPU→CPU→GPU round-trip per position in compact functions (N separate memcpys per head per layer)
  3. No synchronization batching - each position triggers immediate GPU sync
- **Solution**:
  1. Added adaptive compression trigger: `should_compress(cur_pos, committed)` now only compresses when `committed > kv_budget`, avoiding unnecessary compression
  2. Optimized `compact_kv_head_positions` and `compact_tria_k_pre_rope` to batch GPU transfers: read all kept positions into single buffer, compact on CPU, write back in batch with single synchronize
  3. Reduced GPU round-trips from O(N × K × L) to O(1) per compression (where N=positions, K=heads, L=layers)
- **Files changed**:
  - `dflash/src/triattention_runner.h` (adaptive should_compress with committed check)
  - `dflash/src/qwen35/spec_decode.cpp` (updated should_compress call)
  - `dflash/src/qwen35/qwen35_backend.cpp` (updated should_compress call)
  - `dflash/src/triattention_compress.cpp` (batched GPU transfers in both compact functions)
- **Learnings:**
  - GPU memcpy per-position is extremely expensive; always batch transfers and synchronize once at the end
  - TriAttention compression is only beneficial when KV cache exceeds the budget threshold
  - The GPU→CPU→GPU round-trip overhead dominates when compression triggers early/frequently
- **Acceptance criteria status**:
  - ✅ Root cause found (compression too early, GPU transfer overhead)
  - ✅ Adaptive trigger implemented (only compress when above kv_budget)
  - ✅ GPU transfer batching implemented (single buffer per compression cycle)
  - ⏳ Performance verification pending (need to run test_dflash + TriAttention benchmark)

---

## [2026-05-22] - lucebox-hub-gfx1151-wuj
- **Issue**: TriAttention stats file head_dim mismatch - `qwen3.5-27b.bin` had `head_dim=24` (corrupted/wrong), but Qwen3.6-27B has `head_dim=256`, `partial_rotary_factor=0.25`, `rotary_dim=64`
- **Root cause**: Stats file was generated with incorrect dimensions, and code hardcoded `head_dim=64`
- **Solution**: 
  1. Generated new `qwen3.6-27b.bin` stats file with correct dimensions (head_dim=64, freq_count=32, 16 layers, 48 heads, 4 KV heads)
  2. Updated stats path defaults in `triattention_runner.cpp` and `test_triattention.c` to use `qwen3.6-27b.bin`
  3. Updated `CLAUDE.md` docs to reference new stats path
- **Verification**: 
  - TriAttention compression now works without crashes
  - Compressed 1016 → 532 positions (52.4% kept) at kv_budget=128
  - Compression overhead ~900ms (CPU-side bf16→f32 conversion + scoring)
- **Performance note**: Compression overhead dominates any KV cache speedup on this config. Compression runs on CPU (GPU→CPU copy, bf16→f32 conversion, scoring). Need GPU-resident compression for speedup.
- **Files changed**:
  - `dflash/deps/llama.cpp/triattention/stats/qwen3.6-27b.bin` (new)
  - `dflash/src/triattention_runner.cpp` (stats path)
  - `dflash/test/test_triattention.c` (stats path)
  - `dflash/CLAUDE.md` (stats path)
- **Learnings:**
  - Qwen3.6-27B has `head_dim=256` but `partial_rotary_factor=0.25` → only first 64 dims are rotated
  - TriAttention scoring operates on RoPE dimensions only (first `rotary_dim` elements)
  - Stats format: `.bin` (TRIA binary) not `.pt` (PyTorch) - needs converter
  - Stats files at `dflash/deps/llama.cpp/triattention/stats/` with naming `qwen3.X-27b.bin`
- **Acceptance criteria status**:
  - ✅ test_dflash + TriAttention (kv_budget=512) doesn't crash
  - ✅ Compression triggers (log shows [TriAttention] compression)
  - ❌ Speedup >= 20% - NOT ACHIEVED (compression overhead ~900ms on CPU, need GPU-resident)

---

## [2026-05-22] - lucebox-hub-gfx1151-uem
- Already implemented: `ggml_backend_tensor_set_async` replaces `ggml_backend_tensor_set`
- BATCH_SIZE increased to 16 (from original 2)
- Per-batch `ggml_backend_synchronize()` ensures async copies complete before buffer reuse
- Ring buffer with independent host buffers prevents race conditions
- Files changed: `dflash/src/qwen35/gguf_target_loader.cpp` (already modified)
- **Learnings:**
  - The async tensor upload was already implemented in the codebase (likely from previous iteration)
  - Key pattern: mmap → memcpy to ring buffer → async GPU copy → synchronize per batch
  - This avoids CPU-GPU sync points for every tensor, instead syncing only after BATCH_SIZE tensors

---
