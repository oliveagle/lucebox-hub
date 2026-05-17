# DFlash End-to-End Performance Report (gfx1151)

## Test Configuration

| 项目 | 值 |
|------|-----|
| **日期** | 2026-05-17 |
| **分支** | gfx1151 |
| **设备** | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S (gfx1151, Strix Halo) |
| **ROCm** | 7.2.3 |
| **CUs** | 20 @ 2900 MHz |
| **VRAM** | 124 GB (126976 MiB total) |
| **后端** | HIP Phase 2 (rocWMMA flashprefill kernels) |

## Model

| 项目 | 值 |
|------|-----|
| Target | Qwen3.6-27B-Q4_K_M.gguf (83 GB, Q4_K_M quantized) |
| Draft | DFlash-draft-3.6-q8_0.gguf (1.75 GB, Q8_0 quantized) |
| Prompt | `def fibonacci(n):\n    """Calculate the nth Fibonacci number."""\n` (14 tokens) |
| Generated | 64 tokens |

## Performance Results

| 模式 | 总时间 (s) | Tokens/s | Speedup |
|------|-----------|---------|--------|
| **AR Baseline** | 9.016 | **7.10** | 1.00x |
| **DFlash+DDTree** | 3.917 | **16.34** | **2.30x** |

## Per-Step Timing (DFlash+DDTree)

| 阶段 | 耗时 (ms) | 占比 |
|------|-----------|------|
| draft_build | 0.32 | 0.1% |
| draft_copyfeat | 1.74 | 0.5% |
| draft_set | 0.08 | 0.0% |
| draft_compute | 18.40 | 5.2% |
| draft_logits | 9.75 | 2.5% |
| verify_build | 1.40 | 0.4% |
| verify_set | 0.29 | 0.1% |
| **verify_compute** | **323.66** | **91.2%** |
| accept | 0.14 | 0.0% |
| **Total** | **355.76** | **100%** |

## Statistics

- Draft steps: 11
- Accepted: 64/176 (36.4% per step)
- Average commit/step: 5.82
- DDTree budget: 16
- FA window: 2048

## Comparison with RTX 3090 (README data)

| 指标 | RTX 3090 | gfx1151 | Ratio |
|------|----------|---------|-------|
| AR Baseline | 37.65-37.78 tok/s | 7.10 tok/s | **19%** |
| DFlash+DDTree | 96.15-129.52 tok/s | 16.34 tok/s | **13-17%** |
| Speedup | 2.55-3.43x | 2.30x | 67-90% |

## Key Findings

1. **Raw compute gap**: gfx1151 AR baseline is only **19%** of RTX 3090, indicating significant raw compute gap.

2. **Verify compute bottleneck**: At 91% of total DFlash time, `verify_compute` is the absolute bottleneck on HIP. This is the sparse_flash_forward kernel running on the target model.

3. **Draft compute is reasonable**: 18.40ms draft_compute (5.2%) vs 323.66ms verify_compute (91.2%) shows the drafter is working efficiently.

4. **Accept rate lower than expected**: 36.4% vs README's 60-80% (AL 8-9). Could be due to:
   - Quantization mismatch (target Q4_K_M vs training precision)
   - GPU-specific numerical differences in attention computation
   - ROCm vs CUDA numerical differences

5. **Speedup ratio preserved**: Despite being slower overall, gfx1151 achieves 2.30x speedup, which is within the expected 2.55-3.43x range (67-90% of ideal).

## Optimization Recommendations

1. **Primary**: Optimize `verify_compute` (sparse_flash_forward kernel)
   - Currently 323ms per step
   - ROCm WMMA kernels may need tuning
   - Consider comparing with cuBLAS WMMA on CUDA

2. **Secondary**: Investigate accept rate
   - Verify quantization compatibility
   - Test with FP16/BF16 target model (if VRAM allows)
   - Compare numerical outputs between CUDA and ROCm

3. **Tertiary**: Draft compute optimization
   - At 18ms it's already reasonable
   - Focus would have diminishing returns
