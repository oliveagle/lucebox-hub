# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

### HIP tensor upload optimization pattern
- Use `ggml_backend_tensor_set_async` instead of `ggml_backend_tensor_set` for HIP copies
- Ring buffer with BATCH_SIZE >= 8 to queue multiple HIP copies before synchronize
- Pattern: mmap -> ring buffer (memcpy) -> async GPU copy (ggml_backend_tensor_set_async) -> ggml_backend_synchronize
- BATCH_SIZE=16 works well on gfx1151 (128GB memory, no pressure on host buffer)

---

## 2026-05-22 - lucebox-hub-gfx1151-b7z
- Implemented hipMemcpy performance optimization in gguf_target_loader.cpp
- Changed `ggml_backend_tensor_set` → `ggml_backend_tensor_set_async` (line 671)
- Increased `BATCH_SIZE` from 2 → 16 (line 602)
- Removed redundant `host_buf` allocation (was allocated then immediately freed — dead code)
- Ring buffer logic cleaned up and clarified: each batch copies mmap data to ring buffers, submits async GPU copies, then synchronizes
- Build verified: dflash27b, test_dflash, test_generate all compile successfully
- Files changed: `dflash/src/qwen35/gguf_target_loader.cpp`
- **Learnings:**
  - `ggml_backend_tensor_set_async` falls back to synchronous if `backend->iface.set_tensor_async` is NULL, so the code is safe even if the HIP backend doesn't implement the async path
  - The old code had a dead allocation: `host_buf` was malloc'd then immediately free'd after ring buffer allocation, adding ~2x max_tensor_size memory waste
  - BATCH_SIZE=2 meant ~850 sync points for ~1700 tensors; BATCH_SIZE=16 reduces to ~106 sync points — 8x fewer CPU-GPU synchronizations
