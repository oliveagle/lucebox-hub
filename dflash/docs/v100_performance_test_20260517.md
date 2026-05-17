# V100 (sm_70) DFlash Performance Test Report

**Date**: 2026-05-17
**GPU**: Tesla V100 32GB (sm_70)
**Model**: Qwen3.6-27B Q4_K_M + DFlash Draft Q8_0
**Test**: HumanEval 10 prompts, n_gen=128, DDTree budget=22

## Build Configuration

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
```

- V100 uses F16 WMMA kernels (BF16 draft → FP16 at load)
- BSA disabled (requires sm_80+)

## Results Summary

| Metric | Value |
|--------|-------|
| **Mean Decode Speed** | **38.81 tok/s** |
| **Mean Acceptance Length** | 6.05 |
| **Mean Accept Rate** | 37.8% |
| **AL Range** | 4.74 - 8.00 |
| **Speed Range** | 30.1 - 51.0 tok/s |

## Per-Prompt Breakdown

| Prompt | Steps | AL | Accept% | Decode (tok/s) |
|--------|-------|-------|---------|----------------|
| has_close_elements | 16 | 8.00 | 50.0 | 50.66 |
| separate_paren_groups | 16 | 8.00 | 50.0 | 51.00 |
| truncate_number | 27 | 4.74 | 29.6 | 30.09 |
| below_zero | 20 | 6.40 | 40.0 | 41.34 |
| mean_absolute_deviation | 23 | 5.57 | 34.8 | 36.02 |
| intersperse | 26 | 4.92 | 30.8 | 31.38 |
| parse_nested_parens | 24 | 5.33 | 33.3 | 33.86 |
| filter_by_substring | 25 | 5.12 | 32.0 | 33.07 |
| sum_product | 26 | 4.92 | 30.8 | 31.77 |
| rolling_max | 17 | 7.53 | 47.1 | 48.90 |

## Status

✅ All 10 HumanEval prompts completed successfully
✅ No crashes or precision errors
✅ DDTree speculative decoding working correctly
✅ sm_70 WMMA kernels operational

## V100 vs RTX 3090 Comparison (Same Qwen3.6 Target)

| Metric | V100 (sm_70) | RTX 3090 (sm_86) | Ratio |
|--------|:-----------:|:----------------:|:-----:|
| Target | Qwen3.6-27B Q4_K_M | Qwen3.6-27B Q4_K_M | - |
| Draft | dflash-draft-3.6-q8_0.gguf | z-lab/Qwen3.6-27B-DFlash safetensors | - |
| **Mean tok/s** | **38.81** | **77.77** | **0.50×** |
| **Accept Length** | **6.05** | **5.05** | **1.20×** |
| **Accept Rate** | **37.8%** | **32.3%** | **1.17×** |

### Analysis

- V100 decode speed is ~50% of RTX 3090 on the same Qwen3.6 target
- V100 actually achieves **higher** acceptance rate (37.8% vs 32.3%) and AL (6.05 vs 5.05) than 3090 with matched 3.6 draft
- The raw throughput gap is due to sm_70 (Volta F16 WMMA) vs sm_86 (Ampere native BF16 WMMA) architecture difference
- V100 has no BSA support (requires sm_80+)

## Notes

- Using Lucebox Q8_0 GGUF draft for Qwen3.6 (matched with target)
- Qwen3.6 target generally has lower acceptance rates than Qwen3.5 target (see README reference table)
