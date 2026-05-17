# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Warp-based boundary check**: All `block_select_kernel` variants (f16, sm_80+, sm_6x, HIP) must use `(n <= m) && (n < N)` for bounds checking. The `n < N` check is mandatory because `N` can be less than the number of iterations in the warp loop.
- **Output buffer init**: Always initialize `dIdx` with `0xFF` (-1 sentinel) and `dCnt` with `0x00` before kernel launch.
- **Multi-arch consistency**: All kernel variants (flashprefill_f16.cu, flashprefill_kernels.cu, flashprefill_scalar.cu, flashprefill_kernels.hip.cu) must be updated together for boundary check changes.

---

## 2026-05-17 - dflash-v100-flashprefill-optimization-v3i.2.2

### What was implemented
Added complete boundary checks and output buffer initialization across all kernel variants:
1. Added `(n < N)` boundary check to Pass 1 and Pass 2 loops in all kernel variants (f16, sm_80+, sm_6x, HIP)
2. Added `cudaMemset` initialization for `dIdx` (-1 sentinel) and `dCnt` (zero) in all 3 dispatch paths (bf16, f16, f16_pascal)

### Files changed
- `src/flashprefill_kernels.cu`: Added `(n < N)` boundary check in Pass 1 (line 987) and Pass 2 (line 1001)
- `src/flashprefill_scalar.cu`: Added `(n < N)` boundary check in Pass 1 (line 452) and Pass 2 (line 467)
- `src/flashprefill_kernels.hip.cu`: Added `(n < N)` boundary check in Pass 1 (line 653) and Pass 2 (line 665)
- `src/flashprefill.cpp`: Added cudaMemset initialization for dIdx/dCnt in 3 dispatch paths (lines 276, 426, 559)

### Learnings
- **Multi-arch consistency risk**: The same kernel logic exists in 4 different files for different architectures. Changes made in one file must be propagated to all others. The boundary check was already in `flashprefill_f16.cu` from v3i.2.1 but was missing in the other 3 variants.
- **cudaMemset for sentinel values**: Using `0xFF` memset pattern efficiently initializes int32_t buffers to -1 (all bits set), which serves as a sentinel value for invalid block indices. This is a common GPU initialization pattern.

---

## 2026-05-17 - dflash-v100-flashprefill-optimization-v3i.2.1

### What was implemented
Fixed a memory access bug in `block_select_kernel` (flashprefill_f16.cu) that caused crashes for small sequence lengths (S=320, 350).

### Files changed
- `src/flashprefill_f16.cu`: Added `n < N` boundary check in Pass 1 (line 629) and Pass 2 (line 643)
- `test/bench_flashprefill_e2e.cpp`: Fixed `ms` variable scope bug (moved declaration earlier) and added S=320, 350 to test contexts

### Learnings
- **CUDA ternary short-circuit**: The expression `valid ? sp[n*s_n] : NEG_INF` evaluates both branches on GPU! The memory access happens regardless of the predicate. Must explicitly check bounds before accessing.
- **Small M edge case**: For S=320 with BLOCK=128, M=N=3. When iterating `n_base` from 0 to `m` in steps of 32, `n = n_base + lane` can exceed N (e.g., for m=2, n_base=0, lane=3 gives n=3 which is >= N=3).
- **Test fix**: The original crash was actually due to a variable scope bug in bench_flashprefill_e2e.cpp (line 219: `float ms;` declared after being used on line 216).

### Pattern discovered
**Warp-based boundary checking pattern**: When iterating over `n_base` in chunks of warp_size (32), always add explicit bounds checking:
```cuda
for (int n_base = 0; n_base <= m; n_base += 32) {
    int n = n_base + lane;
    bool valid = (n <= m) && (n < N);  // ALWAYS check n < N!
    // Use valid to predicate operations
}
```

---
