# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-23] - lucebox-hub-gfx1151-z4h

### What was implemented
- Created 64K context TriAttention test scripts (`bench_64k_triattention.sh`, `test_triattention_64k.py`)
- Generated 65536 token test prompt
- Investigated DFlash test harness crash

### Files changed
- `dflash/scripts/generate_long_prompt.py` - Already supports 64K (no changes needed)
- `dflash/scripts/bench_64k_triattention.sh` - Created
- `dflash/test_triattention_64k.py` - Created
- `dflash/docs/triattention_64k_test_findings.md` - Created

### Learnings
- **test_dflash crashes during tensor upload**: The HIP backend's `ggml_backend_cuda_set_tensor_async()` fails at `cudaMemcpyAsync()` when uploading `output.weight` (994.63 MiB)
- **Previous success documented**: `docs/dflash_triattention_benchmark_20260520.md` shows TriAttention worked on gfx1151 with +137% speedup (7.62 → 18.09 tok/s)
- **Environment**: 126GB VRAM available, crash not related to OOM
- **Possible cause**: HIP compatibility layer issue or ROCm version mismatch

---

