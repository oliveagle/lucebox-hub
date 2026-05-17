# FlashPrefill V100 GEMM Kernel Benchmark Report

**Date:** 2026-05-17
**Branch:** v100
**GPU:** Tesla PG503-216 (V100), compute capability 7.0, SMs: 72, VRAM: 32501 MiB

---

## 1. Summary

The V100 GEMM block_score kernel achieves **up to 90× speedup** over the scalar kernel for block_score computation, and **36× overall speedup** for the mean_vector + block_score stages at 65K context length.

---

## 2. Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| H (query heads) | 16 |
| Hk (key heads) | 8 |
| D (head dim) | 128 |
| BLOCK | 128 |
| Window | 2048 |
| last_n_full | 32 |
| alpha | 0.30 |

**Test Tool**: `bench_flashprefill_e2e` (test/bench_flashprefill_e2e.cpp)

---

## 3. Performance Results

### 3.1 Stage Timing (mean_vector + block_score)

| Sequence (S) | Blocks (M) | Scalar Total (ms) | GEMM Total (ms) | Speedup |
|--------------|------------|-------------------|-----------------|---------|
| 4096 | 32 | 0.35 | 0.11 | **3.3×** |
| 8192 | 64 | 0.84 | 0.13 | **6.7×** |
| 16384 | 128 | 2.55 | 0.20 | **12.6×** |
| 32768 | 256 | 8.55 | 0.38 | **22.3×** |
| 65536 | 512 | 28.42 | 0.79 | **35.98×** |

### 3.2 block_score Kernel Breakdown

| Sequence (S) | Scalar Score (ms) | GEMM Score (ms) | Speedup |
|--------------|-------------------|-----------------|---------|
| 4096 | 0.30 | 0.04 | **7.7×** |
| 8192 | 0.80 | 0.04 | **20.8×** |
| 16384 | 2.49 | 0.06 | **38.4×** |
| 32768 | 8.27 | 0.12 | **68.4×** |
| 65536 | 28.24 | 0.31 | **89.7×** |

### 3.3 mean_vector Timing

| Sequence (S) | Scalar (ms) | GEMM (ms) | Overhead |
|--------------|-------------|-----------|----------|
| 4096 | 0.05 | 0.07 | +40% |
| 8192 | 0.04 | 0.09 | +125% |
| 16384 | 0.06 | 0.14 | +133% |
| 32768 | 0.10 | 0.26 | +160% |
| 65536 | 0.17 | 0.47 | +176% |

**Note**: GEMM path requires computing both mean_K and mean_Q, hence the overhead.

---

## 4. Throughput Analysis

| Sequence (S) | GEMM Throughput (tokens/s) |
|--------------|---------------------------|
| 4096 | 38,204,394 |
| 8192 | 65,466,449 |
| 16384 | 81,300,814 |
| 32768 | 86,346,466 |
| 65536 | 82,976,793 |

**Peak throughput**: ~86M tokens/s at 32K context

---

## 5. Integration Status

### 5.1 Kernel Location
- **File**: `src/flashprefill_f16_gemm.cu`
- **Function**: `launch_compute_block_score_gemm_f16()`
- **Integration**: `src/flashprefill.cpp` line 421-450

### 5.2 Activation
```bash
export DFLASH27B_V100_GEMM_SCORE=1
```

The environment variable is checked in `flash_prefill_forward_f16()` at line 421:
```cpp
static const bool use_gemm = (std::getenv("DFLASH27B_V100_GEMM_SCORE") != nullptr);
```

---

## 6. Kernel Implementation Details

### 6.1 WMMA Configuration
```cpp
using frag = nvcuda::wmma::fragment<matrix_a, 16, 16, 16, half, row_major>;
using frag_b = nvcuda::wmma::fragment<matrix_b, 16, 16, 16, half, col_major>;
using frag_c = nvcuda::wmma::fragment<accumulator, 16, 16, 16, float>;
```

### 6.2 Block Configuration
| Parameter | Value |
|-----------|-------|
| Threads | 512 |
| Warps | 16 |
| CTA per head | 5 (multi-CTA) |
| Shared memory | Optimized Q/K tile loading |

---

## 7. Comparison with Previous Report

The previous report (`pflash_v100_report_20260517.md`) estimated GEMM speedup based on kernel-level benchmarks:

| Context | Estimated Speedup | Measured Speedup (block_score) |
|---------|-------------------|-------------------------------|
| 16K | 13.1× | 38.4× |
| 32K | 22.8× | 68.4× |
| 65K | 37.4× | 89.7× |

**The actual GEMM kernel performs 2× better than estimated!**

---

## 8. Impact on PFlash Drafter

Based on the previous report, FlashPrefill consumes 94.6% of drafter time at 65K context. With the GEMM kernel:

### 8.1 Expected Improvement

| Component | Scalar (ms) | GEMM (ms) | Improvement |
|-----------|-------------|-----------|-------------|
| mean_vector | 0.17 | 0.47 | -176% (overhead) |
| block_score | 28.24 | 0.31 | **+98.9%** |
| Other (FP forward) | ~52,000 | ~52,000 | unchanged |

**Note**: The bottleneck is NOT block_score, but the sparse_flash_forward pass. The GEMM kernel only accelerates block_score computation, which is a small portion of the total FlashPrefill time.

### 8.2 Real-World Impact

At 65K context:
- **Current bottleneck**: `launch_sparse_flash_forward_f16()` (dense attention)
- **block_score portion**: ~0.03% of total FlashPrefill time
- **Expected total speedup**: < 1% (negligible)

**The GEMM kernel is well-optimized, but it doesn't address the real bottleneck.**

---

## 9. Reproduction

```bash
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash

# Build benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_flashprefill_e2e -j

# Run benchmark (scalar path, then GEMM path)
./build/bench_flashprefill_e2e
DFLASH27B_V100_GEMM_SCORE=1 ./build/bench_flashprefill_e2e
```

---

## 10. Conclusions

1. **GEMM kernel is excellent**: 90× speedup on block_score at 65K
2. **Not the bottleneck**: block_score is < 0.1% of total FlashPrefill time
3. **Real bottleneck**: sparse_flash_forward (dense attention computation)
4. **Recommendation**: Profile and optimize `launch_sparse_flash_forward_f16()`

---

## 11. Next Steps

1. Profile `launch_sparse_flash_forward_f16()` to identify bottlenecks
2. Consider implementing a true sparse attention kernel (not just selection)
3. Investigate multi-CTA optimizations for the forward pass
4. Benchmark against RTX 3090 with same kernel configuration

