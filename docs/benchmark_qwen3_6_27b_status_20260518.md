# vLLM + TriAttention Benchmark Status Report

**Task**: lucebox-hub-gfx1151-z0k.1
**Date**: 2026-05-18
**Model**: Qwen3.6-27B
**GPU**: AMD Radeon 8060S (gfx1151)

## Executive Summary

**BLOCKED**: Cannot run vLLM + TriAttention benchmark due to PyTorch/ROCm compatibility issues.

## Environment Details

### Hardware
- **GPU**: AMD Radeon 8060S (gfx1151)
- **Architecture**: Strix Halo APU
- **VRAM**: 127 GB total (~21 GB used, ~106 GB available)
- **Compute Units**: 40 @ 2900 MHz
- **ROCm**: 7.2.3 (confirmed working via rocminfo)

### Software

| Component | Version | Status |
|-----------|---------|--------|
| ROCm | 7.2.3 | Working (hipcc, rocminfo OK) |
| ROCm PyTorch (therock venv) | 2.10.0+rocm7.0 | GPU detection OK, GPU ops SEGFAULT |
| CUDA PyTorch (.venv) | 2.11.0+cu130 | Incompatible with AMD GPU |
| vLLM (therock) | 0.19.1rc1.dev335 | Present but untested |
| vLLM (.venv) | 0.21.0 | CUDA-dependent, unusable |
| TriAttention | 0.2.0 | Installed |

### Model Files
- **Qwen3.6-27B HF**: `/mnt/eaget-4tb/modelscope_models/Qwen/Qwen3___6-27B` ✓ Available
- **Stats File**: `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt` ✓ Available

## Issues Encountered

### 1. Primary Issue: ROCm PyTorch GPU Operations Segfault

**Symptoms**:
```python
import torch
# GPU detection works
torch.cuda.device_count()  # Returns 1
torch.cuda.get_device_name(0)  # Returns "AMD Radeon 8060S"

# GPU operations crash
x = torch.zeros(100, device='cuda')  # Segmentation fault (core dumped)
```

**Attempted Solutions**:
- Cleaned LD_LIBRARY_PATH (removed CUDA paths)
- Different tensor sizes and operations
- All result in segfault

**Likely Causes**:
- ROCm PyTorch 2.10.0+rocm7.0 vs ROCm 7.2.3 version mismatch
- Kernel driver incompatibility
- Missing or conflicting runtime libraries

### 2. Secondary Issue: CUDA PyTorch Incompatible

**Symptoms**:
```python
# CUDA PyTorch in .venv
RuntimeError: The NVIDIA driver on your system is too old (found version 12020)
```

**Cause**: CUDA-compiled PyTorch cannot use AMD GPU

### 3. llama.cpp Build Failure

**Symptoms**:
```bash
cd dflash/deps/llama.cpp/build-hip
make llama-server
# Error: HIP/CUDA header conflicts in fattn-sparse.cu.o
```

**Cause**: Mixed CUDA/HIP headers causing compilation errors

## Working Components

Despite the issues, these components are confirmed working:

1. **ROCm Toolchain**:
   ```bash
   hipcc --version  # HIP version: 7.2.53211
   rocminfo        # Shows GPU details correctly
   rocm-smi        # Shows GPU status
   ```

2. **PFlash Benchmarks** (from other project files):
   - Working HIP Phase 2 benchmarks documented
   - FlashPrefill kernels functional
   - VRAM tracking working

3. **Project Infrastructure**:
   - Virtual environments set up
   - TriAttention installed
   - Model files downloaded
   - Stats file generated

## TriAttention Support Status

Per TriAttention README:

| Platform | Support | Notes |
|----------|---------|-------|
| vLLM (CUDA) | ✓ Official | Tested on NVIDIA GPUs |
| SGLang | ✓ Official | Added 2026-04-21 |
| llama.cpp (HIP/ROCm) | ✓ Community | @domvox port, AMD GPU support |
| MLX (Apple) | ✓ Community | @DeadByDawn101 port |

**Key Finding**: TriAttention has official community support for AMD GPUs via llama.cpp, but build is failing.

## Alternative Approaches

### Option 1: Fix ROCm PyTorch Segfault (High Effort)

**Steps**:
1. Investigate segfault root cause (gdb, strace)
2. Check kernel driver compatibility
3. Try different ROCm PyTorch versions
4. Reinstall ROCm SDK cleanly

**Estimated Time**: 4-8 hours

**Probability of Success**: Medium

### Option 2: Use SGLang Instead of vLLM

**Rationale**:
- TriAttention has official SGLang support
- SGLang may have better ROCm support
- Different architecture might avoid current issues

**Steps**:
1. Install SGLang in therock environment
2. Test basic GPU operations
3. Run TriAttention benchmarks

**Estimated Time**: 2-4 hours

**Probability of Success**: Medium

### Option 3: Fix llama.cpp Build

**Rationale**:
- TriAttention has community AMD support via llama.cpp
- Avoids vLLM dependency issues
- Direct HIP integration

**Steps**:
1. Fix HIP/CUDA header conflicts
2. Build llama-server with ROCm
3. Test with GGUF model

**Estimated Time**: 2-6 hours

**Probability of Success**: Low-Medium (build issues persist)

### Option 4: CPU Testing (Methodology Validation)

**Rationale**:
- Establish benchmark methodology
- Test TriAttention integration
- Document process for future GPU testing

**Steps**:
1. Test with Qwen3-8B on CPU
2. Run reduced benchmarks
3. Validate metrics collection
4. Prepare for GPU testing

**Estimated Time**: 1-2 hours

**Probability of Success**: High (but slow)

### Option 5: Document and Defer

**Rationale**:
- Current environment has fundamental issues
- ROCm/PyTorch/vLLM integration is complex
- Better to fix environment first

**Outcome**:
- Comprehensive documentation
- Clear next steps
- Environment fix plan

## Recommended Path Forward

Given the complexity and time constraints, I recommend:

### Immediate (Today)
1. **Document findings** ✓ (This report)
2. **Create environment fix plan**
3. **Consider CPU validation** (Option 4)

### Short-term (This Week)
1. **Fix ROCm PyTorch** or **Try SGLang** (Options 1-2)
2. **Reattempt llama.cpp build** (Option 3)
3. **Environment cleanup**

### Long-term
1. **Establish stable ROCm PyTorch environment**
2. **Create benchmark automation scripts**
3. **Document ROCm setup for AMD GPUs**

## Files Created

- `docs/benchmark_qwen3_6_27b_plan_20260518.md` - Benchmark plan
- `docs/benchmark_qwen3_6_27b_status_20260518.md` - This report

## Metrics to Collect (When Environment is Fixed)

| Metric | Unit | Target |
|--------|------|--------|
| Prefill Speed | tokens/s | TBD |
| Decode Speed | tokens/s | TBD |
| First Token Latency | ms | TBD |
| KV Memory | GB | TBD |
| Compression Ratio | % | ~10-30% (budget 2048-3072) |

## Test Scenarios

| Scenario | Context | Output | Priority |
|----------|---------|--------|----------|
| Short | 4,096 | 512 | High |
| Medium | 8,192 | 512 | Medium |
| Long | 16,384 | 512 | Low |

## Conclusion

The vLLM + TriAttention benchmark for Qwen3.6-27B is **blocked** by environment issues. The hardware (AMD Radeon 8060S) and ROCm installation are functional, but PyTorch GPU operations fail with segfaults.

**Recommendation**: Fix the ROCm PyTorch environment before attempting benchmarks. Alternative paths include SGLang or fixing the llama.cpp build.

---

**Next Action**: Environment fix or CPU validation test
**Blocker**: ROCm PyTorch GPU operations segfault
**Est. Resolution Time**: 4-8 hours (environment fix) or 1-2 hours (CPU validation)
