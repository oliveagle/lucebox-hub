# PFlash V100 性能基准测试报告

**日期**: 2026-05-17
**GPU**: Tesla PG503-216 (V100), SM70, 72 SMs, 32GB HBM2
**模型**: Qwen3-0.6B-BF16.gguf (Drafter)
**配置**: keep_ratio=0.10, lookahead=8, chunk=32, pool=13

---

## 1. 测试目标

验证 PFlash 在 V100 上处理 65K tokens 是否能在 **20秒内完成**。

---

## 2. E2E 性能结果

### Scalar Kernel (默认)

| Context | Time (s) | Kept | Ratio | tok/s |
|---------|----------|------|-------|-------|
| 4096    | 0.653    | 1024 | 0.2500 | 6273 |
| 16384   | 2.288    | 1632 | 0.0996 | 7160 |
| 32768   | 5.248    | 3264 | 0.0996 | 6241 |
| **65536** | **12.088** | 6528 | 0.0996 | **5424** |

### GEMM Kernel (DFLASH27B_V100_GEMM_SCORE=1)

| Context | Time (s) | Kept | Ratio | tok/s | Speedup |
|---------|----------|------|-------|-------|--------|
| 4096    | 0.648    | 1024 | 0.2500 | 6318 | 1.01× |
| 16384   | 2.223    | 1632 | 0.0996 | 7371 | 1.03× |
| 32768   | 4.824    | 3264 | 0.0996 | 6790 | 1.09× |
| **65536** | **10.169** | 6528 | 0.0996 | **6446** | **1.19×** |

---

## 3. 结论

### ✅ 目标达成

- **65K tokens 处理时间**: 12.09s (Scalar) / 10.17s (GEMM)
- **目标**: < 20s
- **结果**: **超出目标 40-50%**

### 性能分析

1. **GEMM 加速**: 在 65K context 下，GEMM kernel 提供 19% 加速
2. **吞吐量**: 5424-6446 tok/s，远超实时需求
3. **压缩比**: 稳定在 10% (keep_ratio=0.10)

### 与 Micro-Benchmark 对比

`bench_flashprefill_perf` 测量的 block_score 时间:
- 65K Scalar: 28.075ms → E2E tail-score: 0.73s
- 65K GEMM: 0.766ms → E2E tail-score: 0.74s

E2E 中 tail-score 时间相近，主要差异来自 FlashPrefill 阶段:
- Scalar FP: 7.63s → GEMM FP: 5.70s (节省 1.93s)

---

## 4. 复现

```bash
# Scalar kernel
echo "compress 100 8 32 13 /tmp/test_65k.bin" | \
  ./build/pflash_daemon ./models/Qwen3-0.6B-BF16.gguf

# GEMM kernel
DFLASH27B_V100_GEMM_SCORE=1 ./build/pflash_daemon ./models/Qwen3-0.6B-BF16.gguf
```
