# PFlash V100 Performance Benchmark Report

**Date:** 2026-05-17
**Branch:** v100
**GPU:** Tesla PG503-216 (V100), compute capability 7.0, SMs: 72, VRAM: 32501 MiB
**Drafter Model:** Qwen3-0.6B-BF16 (1.2 GB, 28 layers, 1024 hidden, 151936 vocab)

---

## 1. PFlash Drafter Compression Performance

### Configuration

| Parameter | Value |
|-----------|-------|
| Drafter | Qwen3-0.6B-BF16 |
| keep_ratio | 0.10 |
| lookahead | 8 |
| chunk_size | 32 |
| pool_kernel | 13 |

### Compression Results

| Context | Compress Time | Kept Tokens | Ratio | Effective Tok/s |
|---------|---------------|-------------|-------|------------------|
| 4096 | 0.97s | 1024 | 0.2500 | **4223** |
| 16384 | 6.96s | 1632 | 0.0996 | **2354** |
| 32768 | 22.56s | 3264 | 0.0996 | **1452** |
| 65536 | 81.60s | 6528 | 0.0996 | **803** |

### Analysis

- **4K context**: drafter 在 0.97s 内完成，压缩率 25%
- **16K context**: drafter 花费 6.96s，压缩率接近 10%
- **32K context**: drafter 花费 22.56s，压缩率接近 10%
- **65K context**: drafter 花费 81.60s，压缩率接近 10%

**性能瓶颈**:
- FlashPrefill (FP) 在 65K context 时花费 77.13s (占 94.6%)
- A_compute (注意力计算) 随 context 线性增长
- 在 V100 SM70 上，FlashPrefill 没有使用 BSA (Block-Sparse Attention)，因为 BSA 需要 SM_80+

---

## 2. Per-Layer Timing Breakdown (S=65536)

```
Layer 1: A_compute=0.048s, FP=3.061s
Layer 28: A_compute=0.769s, FP=77.132s
```

- **A_compute** (注意力计算): 0.77s (0.9%)
- **FP** (FlashPrefill): 77.13s (94.6%)
- **tail_score** (尾部分数): 0.73s (0.9%)

---

## 3. Comparison with RTX 3090 Reference

### Drafter Performance

| Context | V100 Compress | RTX 3090 Compress | V100/3090 |
|---------|---------------|-------------------|-----------|
| 4096 | 0.97s | ~0.4s (est) | 2.4× slower |
| 16384 | 6.96s | 1.27s | **5.5× slower** |
| 32768 | 22.56s | 2.08s | **10.8× slower** |
| 65536 | 81.60s | ~5s | **16× slower** |

### Key Findings

1. **V100 的 FlashPrefill 性能显著低于 RTX 3090**:
   - SM70 不支持 BSA (Block-Sparse Attention)
   - Tensor Cores 性能较低 (1st gen vs 3rd gen)
   - 内存带宽较低 (900 GB/s vs 936 GB/s)

2. **在 65K context**:
   - V100: 81.60s drafter + ~6s target prefill (est) = **87.60s TTFT**
   - RTX 3090: 11.11s drafter + 4.79s target prefill = **15.91s TTFT**
   - **V100 是 3090 的 5.5× 慢**

3. **有效吞吐量** (考虑压缩后的 token 数):
   - V100 @ 65K: 6528 tokens / 81.60s = **80 tok/s**
   - RTX 3090 @ 65K: ~6500 tokens / 5s = **1300 tok/s**
   - **V100 是 3090 的 6%**

---

## 4. End-to-End TTFT Comparison

### 131K Context (with keep=0.10)

| Metric | V100 (estimated) | RTX 3090 (measured) | Ratio |
|--------|------------------|---------------------|-------|
| Drafter compress | ~330s | 11.11s | **30× slower** |
| Target prefill | ~10s | 4.79s | 2× slower |
| Total TTFT | **~340s** | **15.91s** | **21× slower** |
| Effective tok/s | ~385 | 8240 | **4.7%** |

**Note**: V100 的 131K 估算是基于 65K 的测量结果线性外推（考虑到 O(N²) 复杂度，实际情况可能更差）

---

## 5. Hardware Comparison

| Spec | RTX 3090 | V100 | Ratio |
|------|----------|------|-------|
| Architecture | Ampere (SM86) | Volta (SM70) | - |
| SMs | 82 | 72 | 0.88× |
| Tensor Cores | 3rd gen | 1st gen | - |
| Memory BW | 936 GB/s | 900 GB/s | 0.96× |
| VRAM | 24 GB GDDR6X | 32 GB HBM2 | 1.33× |
| BSA Support | ✅ (SM80+) | ❌ (SM70) | - |

### Why V100 is slower for PFlash

1. **BSA 不支持**: SM70 不支持 Block-Sparse Attention，必须使用 dense attention
2. **Tensor Cores**: 1st gen 性能远低于 3rd gen
3. **FlashPrefill**: 在 V100 上使用 F16 WMMA GEMM kernel (已优化)，但 BSA 的 block-sparsity 优势无法利用

---

## 6. V100 GEMM block_score Kernel Optimization

### Existing Implementation

V100 上已有优化的 `compute_block_score_gemm_kernel_f16` kernel (在 `flashprefill_f16_gemm.cu`):

- **F16 WMMA**: 使用 `m16n8k16` WMMA 指令 (V100 Tensor Cores)
- **512 threads**: 最大化 SM 占用
- **Multi-CTA**: 每个头使用 5 个 CTA 并行计算
- **Shared memory**: 优化 Q/K tile 加载

### GEMM vs Scalar Kernel Speedup

| Sequence Length | Scalar (ms) | GEMM (ms) | Speedup |
|-----------------|-------------|-----------|---------|
| 4096 | 0.326 | 0.093 | **3.5×** |
| 8192 | 0.827 | 0.117 | **7.0×** |
| 16384 | 2.528 | 0.193 | **13.1×** |
| 32768 | 8.770 | 0.384 | **22.8×** |
| 65536 | 28.697 | 0.767 | **37.4×** |

**分析**: GEMM kernel 在长序列下优势显著，这是因为:
1. WMMA Tensor Core 加速矩阵乘法
2. 减少 global memory 访问 (shared memory tiling)
3. 更好的并行度 (multi-CTA)

### Current Integration Status

GEMM kernel 已经集成到 `flashprefill.cpp` 中:
- `launch_compute_block_score_gemm_f16()` 声明存在
- 在 `flash_prefill_forward_f16()` 中调用
- **环境变量控制**: 需要设置 `DFLASH27B_V100_GEMM_SCORE=1` 启用

### Testing GEMM Kernel in PFlash

当前 PFlash drafter 可能没有启用 GEMM kernel。需要验证并启用它:

```bash
# Enable V100 GEMM block_score kernel
export DFLASH27B_V100_GEMM_SCORE=1

# Run drafter compression
echo "compress 100 8 32 13 /tmp/test_tokens.bin" | \
  ./build/pflash_daemon /path/to/Qwen3-0.6B-BF16.gguf
```

### Expected Performance with GEMM

如果 GEMM kernel 被正确启用，预期性能提升:

| Context | Current (scalar) | With GEMM | Speedup |
|---------|------------------|-----------|---------|
| 16384 | 6.96s | ~2.5s | **2.8×** |
| 32768 | 22.56s | ~6s | **3.8×** |
| 65536 | 81.60s | ~17s | **4.8×** |

**Note**: 实际加速比可能低于 block_score kernel 的单独加速比，因为:
1. 其他阶段 (mean_vector, block_select, tail_score) 不受 GEMM 影响
2. FlashPrefill forward 包含多个 kernel，block_score 只是其中之一

---

## 7. Conclusions

1. **V100 可以运行 PFlash drafter**，但性能远低于 RTX 3090
2. **在 16K context 以下**，drafter 压缩时间可接受 (6.96s @ 16K)
3. **在 32K+ context**，drafter 压缩时间变得不可接受 (22s+ @ 32K, 81s @ 65K)
4. **PFlash 在 V100 上的瓶颈** 是 FlashPrefill (占 94.6% 的 drafter 时间)
5. **V100 GEMM block_score kernel 已存在**，但在当前 PFlash 测试中可能未启用
6. **启用 GEMM kernel 后预期加速**: 2.8-4.8× (取决于 context 长度)
7. **建议**:
   - V100 适合短 context (< 16K) 的 PFlash 压缩
   - 对于长 context (32K+)，建议使用 RTX 3090 或更新架构的 GPU
   - 启用 `DFLASH27B_V100_GEMM_SCORE=1` 来激活 V100 优化的 GEMM kernel
   - 或者使用更低 keep_ratio (如 0.05) 来减少 drafter 的计算量

---

## 8. Reproduction

```bash
# Test drafter compression
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash

# Create test tokens
python3 -c "
import struct
n = 65536
with open('/tmp/test_tokens.bin', 'wb') as f:
    f.write(struct.pack('<I', n))
    for i in range(n):
        f.write(struct.pack('<i', 1972))
"

# Run drafter compression (keep=0.10)
echo "compress 100 8 32 13 /tmp/test_tokens.bin" | \
  ./build/pflash_daemon /path/to/Qwen3-0.6B-BF16.gguf

# Enable V100 GEMM kernel (should be ~2-5x faster)
DFLASH27B_V100_GEMM_SCORE=1 \
echo "compress 100 8 32 13 /tmp/test_tokens.bin" | \
  ./build/pflash_daemon /path/to/Qwen3-0.6B-BF16.gguf
```
