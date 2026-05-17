# Block Select Kernel Investigation Report

**Date:** 2026-05-17
**Investigator:** Oliveagle

## Summary

The original bug report stated "block_select 导致崩溃" (block_select causes crash). After investigation, this is **partially incorrect**.

## Findings

### 1. Block Select Kernel - WORKING

- **Verified with benchmark**: block_select kernel works correctly at all tested sizes (320, 350, 4096, 8192, 16384, 32768, 65536)
- **No illegal memory access**: The kernel passes cuda-memcheck at all sizes
- **Correct output**: Indices and counts are computed correctly

### 2. Sparse Flash Forward Kernel - CRASHING

- **Crash point**: `sparse_flash_forward_f16` kernel in `src/flashprefill_f16.cu`
- **Trigger**: Sequence lengths >= 16384 tokens
- **Error**: Illegal memory access detected during cleanup (after forward pass)
- **Location**: Not in block_select, but in the WMMA attention kernel

### 3. Test Results

| S (tokens) | block_select | sparse_flash_forward | Result |
|------------|--------------|----------------------|--------|
| 320        | ✅ OK        | ✅ OK                | PASS   |
| 4096       | ✅ OK        | ✅ OK                | PASS   |
| 8192       | ✅ OK        | ✅ OK                | PASS   |
| 16384      | ✅ OK        | ❌ CRASH             | FAIL   |
| 32768      | ✅ OK        | ❌ CRASH             | FAIL   |
| 65536      | ✅ OK        | ❌ CRASH             | FAIL   |

### 4. Root Cause Analysis

The `sparse_flash_forward_f16` kernel has a memory access issue at large sequence lengths.

**Possible causes:**
1. Out-of-bounds access in K/V tile loading when `block * BLOCK >= seq_len`
2. Incorrect shared memory usage
3. Grid/block configuration issue for large `q_tiles`

## Files Modified

- `test/bench_flashprefill_e2e.cpp`: Enabled block_select in benchmark (was commented out)

## Files to Investigate

- `src/flashprefill_f16.cu`: `sparse_flash_forward_kernel_f16` (lines 210-548)
