# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-23] - lucebox-hub-gfx1151-m7j

- **What was implemented:**
  - Created GPU kernel infrastructure for TriAttention KV compression:
    - `src/triattention_kernels.h` - Shared device code header with `__device__` scoring functions
    - `src/triattention_kernels.cu` - CUDA translation unit with host-side launchers
    - `src/triattention_kernels.cuh` - Header with host-side function declarations
    - `src/triattention_kernels.hip.cu` - HIP/ROCm translation unit
  - Updated `CMakeLists.txt` to include kernel source files (later commented out due to build issues)
  - Restored `triattention_compress.cpp` to working CPU path with improved comments

- **Files changed:**
  - `dflash/src/triattention_kernels.h` (new, 212 lines)
  - `dflash/src/triattention_kernels.cu` (new, 401 lines)
  - `dflash/src/triattention_kernels.cuh` (updated, 134 lines)
  - `dflash/src/triattention_kernels.hip.cu` (new, 450 lines)
  - `dflash/CMakeLists.txt` (added kernel sources, then commented out)
  - `dflash/src/triattention_compress.cpp` (cleaned up, restored CPU path)

- **Learnings:**
  - **Pattern: GPU kernel infrastructure setup** - Created separate files for CUDA/HIP compatibility with shared device code in a `.h` header
  - **Pattern: Backend-specific compilation** - Used CMake `LANGUAGE` property to compile HIP files with the correct compiler
  - **Gotcha: Template kernel instantiation** - Template kernels for different FC values (16, 32, 64, 128) require careful handling to avoid compilation errors
  - **Gotcha: HIP/CUDA API differences** - `hipLaunchKernelGGL` vs `<<<>>>` syntax, `hipStream_t` vs `cudaStream_t`, `hip_bfloat16` vs `__nv_bfloat16`
  - **Gotcha: Build complexity** - GPU kernel code with complex templates and thrust integration is error-prone; better to start with simple CPU path and incrementally add GPU optimization
  - **Current state**: CPU-based TriAttention compression works but has GPU→CPU→GPU round trip bottleneck. GPU kernel infrastructure is in place for future optimization.
  - **Key bottleneck**: The `tria_score_kv_head()` function runs on CPU, processing each (layer, KV head, position) combination. This is the main CPU bottleneck that needs to be moved to GPU.

---
