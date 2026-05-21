# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it should be included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-22] - lucebox-hub-gfx1151-5dq

### Root Cause: gfx1151 APU Host-to-Device Memory Copy is Extremely Slow

The "hang" in test_dflash and test_generate is NOT a hang - it's extremely slow H2D memory transfer on the gfx1151 APU.

**Key Finding**: Radeon 8060S (gfx1151) is an **APU** (integrated GPU) that shares system memory, not a discrete GPU with dedicated VRAM.

- The first tensor `output.weight` (994.63 MiB) takes **more than 12 minutes** to upload
- Total model size is ~15 GiB with 850+ tensors
- At this rate, full model loading would take **3+ hours**

### Timeline of Investigation

1. **Initial observation**: Program hangs after "uploading tensor 1/851: output.weight (994.63 MiB)"
2. **GGML analysis**: Confirmed gfx1151 workarounds already in place (sync memcpy, no hipStreamPerThread)
3. **Direct HIP copy test**: Using hipMemcpyAsync with explicit non-blocking stream - same slow result
4. **Root cause identified**: gfx1151 APU system memory architecture

### Existing Workarounds in GGML

- `ggml_backend_cuda_buffer_set_tensor`: Uses synchronous `cudaMemcpy` (not async)
- `ggml_backend_cuda_buffer_init_tensor`: Skips per-tensor cudaMemset (batched approach)
- `ggml_backend_cuda_buffer_set_tensor_2d`: Uses explicit stream (not hipStreamPerThread)
- CUDA/HIP Graphs: Auto-disabled for gfx1151

### Why All Approaches Are Slow

On an APU, H2D memcpy is copying from system RAM to... system RAM (just different address ranges).
The copy goes through:
1. CPU read from application buffer
2. Write to GPU-accessible memory region
3. Potential cache coherency operations

This is fundamentally slower than discrete GPU H2D transfers over PCIe.

### Recommendations

1. **Use a discrete GPU** for development/testing (e.g., RX 7900 series with dedicated VRAM)
2. **Reduce model size** - Q4_K_M quantization still too large for practical APU use
3. **Consider llama.cpp** - may have better APU optimizations

---
