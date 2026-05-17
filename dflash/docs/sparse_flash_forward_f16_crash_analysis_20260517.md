# Sparse Flash Forward F16 Kernel Crash Analysis

**Date:** 2026-05-17
**Kernel:** `sparse_flash_forward_kernel_f16` in `src/flashprefill_f16.cu`
**Architecture:** Volta (SM_70), V100

## Summary

The F16 WMMA sparse flash forward kernel crashes at sequence lengths >= 4096 with an illegal memory access. The kernel works correctly for S=320 and S=350.

## Crash Details

- **Crash point:** S=4096, block (2, 7, 0), thread (32, 0, 0)
- **Error:** Invalid __global__ read of size 4 bytes
- **Offset:** 0x20f0 in kernel binary
- **Compute-sanitizer output:**
  ```
  Invalid __global__ read of size 4 bytes
  at void dflash27b::flashprefill::sparse_flash_forward_kernel_f16<(int)64, (int)64, (int)128, (int)128>
  by thread (32,0,0) in block (2,7,0)
  Address 0x... is out of bounds
  and is 205 bytes after the nearest allocation at 0x... of size 576 bytes
  ```

## Grid Configuration

For S=4096, H=16, Hk=8, D=128, BLOCK=128, Q_TILE=64:
- M = 4096/128 = 32 blocks
- q_tiles = 64 tiles
- Grid = (64, 16, 1) [q_tiles, batch*n_q_heads, 1]
- Block = (64, 1, 1) [64 threads per block]

Block (2, 7, 0) maps to:
- q_tile_idx = 2
- zh = 7 → b = 0, qh = 7
- q_block_idx = 2 * 64 / 128 = 1
- Thread 32 → wid = 1, lane = 0

## Buffer Sizes

- dCnt: B * M * H = 1 * 32 * 16 = 512 int32_t elements
- dIdx: B * M * M * H = 1 * 32 * 32 * 16 = 16384 int32_t elements

## Suspected Issues

1. **Row state access in warp 1:** The kernel uses 64 threads (2 warps) with WMMA m32n8k16. Each warp maintains row_m/row_l state for 32 rows. Warp 1 accesses row_m[32..63] and row_l[32..63].

2. **Stride calculation:** For S=4096, the stride values are:
   - s_cnt_b = M * H = 32 * 16 = 512
   - s_cnt_m = H = 16
   - s_cnt_h = 1
   
   The counts access at line 292: `counts[b * s_cnt_b + q_block_idx * s_cnt_m + qh * s_cnt_h]`
   = `counts[0 * 512 + 1 * 16 + 7 * 1]` = `counts[23]`, which is valid.

3. **Shared memory layout:**
   - Q_sh: 64 * 128 = 8192 halves = 16384 bytes
   - KV_sh: 64 * 128 = 8192 halves = 16384 bytes
   - P_sh: 64 * 64 = 4096 halves = 8192 bytes
   - row_m: 64 floats = 256 bytes
   - row_l: 64 floats = 256 bytes
   - Total: ~41 KB

## Test Results

| S  | M   | Result |
|----|-----|--------|
| 320 | 3   | PASS   |
| 350 | 3   | PASS   |
| 4096| 32  | CRASH  |
| 8192| 64  | CRASH  |

## Next Steps

1. Add debug output to the kernel to identify exact crash location
2. Compare with BF16 WMMA kernel (sm_80+) which works correctly
3. Check for integer overflow in address calculations at larger S
4. Verify WMMA fragment layout for m32n8k16 matches the assumed indexing

## Files to Investigate

- `src/flashprefill_f16.cu`: Lines 209-543 (sparse_flash_forward_kernel_f16)
- `src/flashprefill_kernels.cu`: BF16 WMMA kernel for comparison
- `test/test_volta_f16.cpp`: E2E test that reproduces the crash
