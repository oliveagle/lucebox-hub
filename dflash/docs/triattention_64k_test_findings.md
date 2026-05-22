# 64K Context TriAttention Performance Test - Findings

## Summary
The task was to validate TriAttention KV compression performance at 64K context length, but the test harness crashes during model loading phase.

## Environment
- **GPU**: Radeon 8060S Graphics (gfx1151, Strix Halo)
- **VRAM**: 126GB
- **System RAM**: 128GB
- **Backend**: ROCm/HIP

## Issue
The `test_dflash` binary crashes with SIGSEGV during tensor upload phase:
```
[loader] allocated ring buffer: 16 x 1042944000 bytes (15914.06 MiB)
[loader] tensor 1/850: output.weight (994.63 MiB)
Segmentation fault (core dumped)
```

### Crash Analysis
1. **Location**: `ggml_backend_cuda_set_tensor_async()` → `cudaMemcpyAsync()`
2. **Tensor**: `output.weight` (994.63 MiB, first tensor uploaded)
3. **Ring buffer**: 16 x ~1GB = ~16GB allocated successfully
4. **GPU memory**: 126GB available, should be sufficient

### Possible Causes
1. **HIP compatibility issue**: `cudaMemcpyAsync` may not be properly mapped to `hipMemcpyAsync`
2. **Memory address issue**: The mmap'd file address + offset might be problematic
3. **Stream issue**: The CUDA/HIP stream might not be properly initialized
4. **ROCm version**: The ROCm 7.2 installation might have compatibility issues

## Previous Successful Test
According to `docs/dflash_triattention_benchmark_20260520.md`, a test was run on the same GPU (gfx1151) and showed:
- **Baseline**: 7.62 tok/s
- **TriAttention**: 18.09 tok/s (+137% improvement)
- **Output consistency**: 100%

This suggests the code worked at some point, and the issue might be:
1. Different model configuration (the previous test used different model paths)
2. ROCm/HIP driver version change
3. Code changes that introduced the crash

## Recommendations
1. **Debug the crash**: Add logging to `ggml_backend_cuda_set_tensor_async()` to understand the failure
2. **Check ROCm version**: Verify ROCm/HIP driver compatibility
3. **Use sync upload**: Try using `ggml_backend_tensor_set()` instead of async
4. **Verify model format**: Ensure GGUF model format is compatible
5. **Reduce batch size**: The ring buffer of 16x max_tensor_size might be too large

## Files Changed
- `dflash/scripts/generate_long_prompt.py` - Already supports 64K token generation
- `dflash/scripts/bench_64k_triattention.sh` - Created for 64K benchmarking
- `dflash/test_triattention_64k.py` - Created for testing

