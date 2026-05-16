# DFlash Benchmark Report - V100 SM70

**Date:** 2026-05-17
**GPU:** Tesla PG503-216 (V100), compute capability 7.0, VRAM: 32501 MiB
**Models:**
- Target: Qwen3.6-27B-Q4_K_M.gguf (14.99 GiB)
- Draft: dflash-draft-3.6-q8_0.gguf (1.75 GiB)

## Configuration

- n_gen: 128
- DDTree budget: 22
- Fast rollback: enabled
- Flash attention window: 2048

## Results (5 HumanEval prompts)

| prompt | AR tok/s | DFlash tok/s | Speedup | AL |
|--------|----------|--------------|---------|-----|
| has_close_elements | 20.43 | 39.73 | **1.94x** | 6.10 |
| separate_paren_groups | 20.46 | 63.99 | **3.13x** | 9.85 |
| truncate_number | 20.54 | 27.06 | **1.32x** | 4.13 |
| below_zero | 20.54 | 39.76 | **1.94x** | 6.10 |
| mean_absolute_deviation | 20.48 | 44.26 | **2.16x** | 6.74 |
| **Average** | **20.49** | **42.96** | **2.10x** | **6.58** |

## Comparison with RTX 3090 (from README)

| Metric | RTX 3090 | V100 | Ratio |
|--------|----------|------|-------|
| AR tok/s | 37.78 | 20.49 | 0.54x |
| DFlash tok/s | 129.52 | 42.96 | 0.33x |
| Speedup | 3.43x | 2.10x | - |

## Analysis

V100 (SM70) shows lower absolute performance compared to RTX 3090 (SM86):
- AR baseline: V100 is ~54% the speed of RTX 3090
- DFlash: V100 is ~33% the speed of RTX 3090
- Speedup ratio: V100 achieves ~61% of RTX 3090's speedup

This is expected due to:
1. Older architecture (SM70 vs SM86)
2. Lower memory bandwidth on V100 (lesser than RTX 3090's 936 GB/s)
3. CUDA cores and tensor core differences

## FlashPrefill Kernel Benchmarks (separate)

GEMM path speedup vs scalar path at different sequence lengths:

| Sequence Length | Scalar (ms) | GEMM (ms) | Speedup |
|-----------------|-------------|-----------|---------|
| 4096 | 0.326 | 0.093 | **3.5x** |
| 8192 | 0.827 | 0.117 | **7.0x** |
| 16384 | 2.528 | 0.193 | **13.1x** |
| 32768 | 8.770 | 0.384 | **22.8x** |
| 65536 | 28.697 | 0.767 | **37.4x** |

The GEMM-based block score kernel shows excellent scaling with sequence length.