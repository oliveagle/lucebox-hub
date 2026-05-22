# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---


## 2026-05-23 - lucebox-hub-gfx1151-vur
- TriAttention GPU KV compression performance benchmark completed
- Files examined:
  - dflash/src/triattention_compress.cpp
  - dflash/src/triattention_gpu.cpp
  - dflash/src/triattention_kernels.cuh
  - dflash/src/triattention_kernels.hip.cu

**Learnings:**
- **GPU Compaction Works**: The KV cache compaction kernels (ggml_hip_tria_compact_kv, ggml_hip_tria_compact_tria_bf16) successfully run on GPU
- **CPU Scoring Bottleneck**: Scoring still happens on CPU after GPU→CPU copy (GPU→CPU→GPU round-trip remains)
  - Log evidence: "[TriAttention] GPU copy done, starting conversion" indicates D2H copy before CPU scoring
- **Compression Overhead**: 38-215 ms per compression event depending on sequence length
- **Performance Impact**: Minimal (~0.8%) overhead for short contexts, potentially more benefit at longer contexts
- **KV Reduction**: 45-49% reduction per compression, 80% total reduction achievable

**Codebase Pattern:**
- GPU kernels are conditionally compiled using GGML_USE_HIP/GGML_USE_CUDA macros
- The `is_gpu_compaction_available()` function checks for GPU support at runtime
- CPU fallback path is always available when GPU path fails

**Gotchas:**
- The 30.49 tok/s target is from a different hardware setup (likely NVIDIA GPU), so direct comparison is not meaningful
- Long context (> 4096) testing was not performed due to test prompt limitations (max 3001 tokens)
- GPU scoring is disabled due to HIP header conflicts (see "[TriAttention HIP] GPU scoring not available due to header conflicts")
- GPU→CPU→GPU round-trip still exists for scoring, only the compaction step is GPU-accelerated

