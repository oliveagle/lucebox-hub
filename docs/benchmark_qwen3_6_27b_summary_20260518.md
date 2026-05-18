# vLLM + TriAttention Benchmark Test - Execution Summary

**Task**: Run vLLM + TriAttention benchmark tests for Qwen3.6-27B
**Date**: 2026-05-18
**Status**: BLOCKED - Environment Issues
**Project**: lucebox-hub-gfx1151

## Task Objectives

1. Start vLLM server with TriAttention for Qwen3.6-27B
2. Run comprehensive benchmark tests (4K, 8K, 16K contexts)
3. Measure: prefill speed, decode speed, first token latency, KV memory usage
4. Generate benchmark report comparing with baseline

## Environment Status

### Hardware
- **GPU**: AMD Radeon 8060S (gfx1151) - 127 GB VRAM
- **ROCm**: 7.2.3 (confirmed working)
- **Status**: GPU detected and operational

### Software Stack

| Component | Status | Notes |
|-----------|--------|-------|
| ROCm Toolchain | ✓ Working | hipcc, rocminfo, rocm-smi OK |
| ROCm PyTorch (therock venv) | ✗ Broken | GPU ops segfault |
| CUDA PyTorch (.venv) | ✗ Incompatible | Cannot use AMD GPU |
| vLLM (therock) | ? Untested | Present but PyTorch broken |
| vLLM (.venv) | ✗ Incompatible | CUDA-dependent |
| TriAttention | ✓ Installed | v0.2.0 |
| Qwen3.6-27B Model | ✓ Available | HF format, 54 GB |
| Stats File | ✓ Generated | 1.6 MB stats.pt |

## Critical Issue: ROCm PyTorch Segfault

**Problem**:
```python
# GPU detection works
torch.cuda.device_count()  # Returns 1
torch.cuda.get_device_name(0)  # Returns "AMD Radeon 8060S"

# GPU operations crash with segfault
x = torch.zeros(100, device='cuda')  # Segmentation fault (core dumped)
```

**Impact**: Cannot run any GPU-based inference or benchmarks.

**Root Cause**: Likely ROCm PyTorch 2.10.0+rocm7.0 vs ROCm 7.2.3 version mismatch or kernel driver incompatibility.

## Investigation Results

### Attempted Solutions

1. **Library Path Cleanup**
   - Removed CUDA paths from LD_LIBRARY_PATH
   - Result: Still segfaults

2. **Environment Testing**
   - Tested with therock venv (ROCm PyTorch)
   - Tested with .venv (CUDA PyTorch)
   - Result: Both unusable for different reasons

3. **llama.cpp Build**
   - Attempted to build llama.cpp with ROCm support
   - Result: HIP/CUDA header conflicts

### Working Components

- ROCm CLI tools (hipcc, rocminfo, rocm-smi)
- Model files and stats file available
- TriAttention installed and configured
- Project infrastructure ready

## Alternative Approaches Evaluated

### Option 1: Fix ROCm PyTorch (High Effort)
- Investigate segfault with gdb/strace
- Try different PyTorch versions
- Reinstall ROCm SDK
- **Est. Time**: 4-8 hours
- **Probability**: Medium

### Option 2: Use SGLang
- TriAttention has official SGLang support
- Different architecture might avoid issues
- **Est. Time**: 2-4 hours
- **Probability**: Medium

### Option 3: Fix llama.cpp Build
- Community TriAttention support for AMD
- Direct HIP integration
- **Est. Time**: 2-6 hours
- **Probability**: Low-Medium

### Option 4: CPU Validation Test
- Validate benchmark methodology
- Test with smaller model on CPU
- **Est. Time**: 1-2 hours
- **Probability**: High

## Documentation Created

1. **`docs/benchmark_qwen3_6_27b_plan_20260518.md`**
   - Detailed benchmark plan
   - Test scenarios and metrics
   - Configuration parameters

2. **`docs/benchmark_qwen3_6_27b_status_20260518.md`**
   - Environment status report
   - Issue analysis
   - Recommended path forward

3. **This file**
   - Execution summary
   - Quick reference

## Recommendations

### Immediate Action
**Priority: Fix environment before proceeding**

The fundamental issue (PyTorch GPU ops) must be resolved before any benchmarks can run.

### Recommended Approach

**Option A: Environment Fix (Long-term solution)**
1. Debug ROCm PyTorch segfault
2. Establish stable ROCm environment
3. Run full benchmarks

**Option B: Alternative Backend (Medium-term)**
1. Try SGLang with TriAttention
2. Test GPU compatibility
3. Run benchmarks if successful

**Option C: CPU Validation (Short-term)**
1. Run reduced benchmarks on CPU
2. Validate methodology
3. Prepare for GPU testing

## Next Steps

1. **Decide on approach** (A/B/C)
2. **Fix environment** or **try alternative**
3. **Re-run benchmarks**
4. **Generate report**

## Files and Locations

- **Model**: `/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3___6-27B`
- **Stats**: `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt`
- **Scripts**: `scripts/run_vllm_with_triattention.sh`
- **Docs**: `docs/benchmark_qwen3_6_27b_*.md`

## Metrics to Collect (When Unblocked)

| Metric | Unit | Tool |
|--------|------|------|
| Prefill Speed | tokens/s | vLLM logs |
| Decode Speed | tokens/s | vLLM logs |
| First Token Latency | ms | vLLM logs |
| KV Memory | GB | rocm-smi |
| Compression Ratio | % | TriAttention stats |

## Conclusion

**The vLLM + TriAttention benchmark for Qwen3.6-27B is currently blocked by PyTorch/ROCm environment issues.**

The hardware and software components are present, but GPU operations fail with segfaults. Environment fix is required before benchmarks can proceed.

---

**Status**: BLOCKED
**Blocker**: ROCm PyTorch GPU operations segfault
**Resolution Path**: Environment fix or alternative backend
**Est. Time**: 2-8 hours depending on approach
