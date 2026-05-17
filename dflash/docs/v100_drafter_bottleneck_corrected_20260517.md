# V100 PFlash Drafter 瓶颈分析 (修正版)

**Date:** 2026-05-17
**GPU:** Tesla V100 (SM70, 72 SMs)
**Model:** Qwen3-0.6B-BF16 (GGUF)

---

## 关键发现

**PFlash drafter 根本不使用优化的 WMMA kernels！**

### 代码路径分析

```
drafter → forward_qwen3_drafter_model()
         → qwen3_graph.cpp:flash_prefill_forward_q8()
         → ggml flash_attn_ext (DENSE CAUSAL ATTENTION)
```

**未被使用**:
- `flash_prefill_forward_f16()` (F16 WMMA + GEMM block_score)
- `flash_prefill_forward_bf16()` (BF16 WMMA + BSA)

### 为什么不使用优化 kernels?

查看 `qwen3_graph.cpp:509-515`:

```cpp
// Drafter uses Q8 quantization, not BF16/F16
const bool use_bf16_fp = (Q_buf.t->type == GGML_TYPE_BF16) && SM80+;  // ❌
const bool use_f16_fp = use_volta_fp && (Q_buf.t->type == GGML_TYPE_F16);  // ❌

// Falls back to:
flash_prefill_forward_q8() → ggml_flash_attn_ext
```

**问题**:
1. V100 是 SM70，不是 SM80+ (BF16 WMMA requires SM80)
2. Drafter 使用 Q8 量化，不是 F16/BF16
3. F16 WMMA kernels 被 **故意禁用** (line 502: "has uninitialized read bug")

---

## 性能分析

### 实测数据 (S=65536)

| Stage | Time | % |
|-------|------|-----|
| A_compute | 0.77s | 0.9% |
| **FP (ggml flash_attn_ext)** | **77.11s** | **94.8%** |
| tail_score | 0.73s | 0.9% |
| **Total** | **81.48s** | **100%** |

### ggml flash_attn_ext 的限制

- **Dense causal attention**: 计算 O(S²) 的注意力矩阵
- **No block sparsity**: 无法利用 BSA (Block-Sparse Attention)
- **No top-K selection**: 每个 token 都关注所有之前的 tokens

---

## 为什么 GEMM 优化无效?

### `bench_flashprefill_e2e` 测的是什么?

```
mean_vector + block_score (仅这两个阶段)
≈ 0.79 ms @ 65K with GEMM
```

### Drafter 实际用的是?

```
ggml flash_attn_ext
≈ 77100 ms @ 65K (97,000× 慢!)
```

**结论**: `bench_flashprefill_e2e` 测试的代码路径和 drafter 使用的代码路径完全不同！

---

## 优化方案 (重新评估)

### 方案 A: 强制启用 F16 WMMA (高风险)

**方法**: 设置 `DFLASH_FP_USE_VOLTA_FP=1`

**问题**:
- 代码注释: "has uninitialized read bug in block_select for certain sequence lengths"
- 可能导致内存损坏或错误结果

**测试**:
```bash
DFLASH_FP_USE_VOLTA_FP=1 ./build/pflash_daemon model.gguf
```

---

### 方案 B: 使用 Ampere GPU (最有效)

**RTX 3090 (SM86)** vs **V100 (SM70)**:

| 特性 | V100 | RTX 3090 |
|------|------|----------|
| 架构 | Volta | Ampere |
| Tensor Cores | 1st gen F16 | 3rd gen BF16 |
| BSA Support | ❌ | ✅ |
| Shared Memory | 48 KB | 164 KB |
| flash_attn_ext | 可能有优化 | 更优化的 kernel |

---

### 方案 C: 降低 keep_ratio (立即可行)

**当前**: keep_ratio=0.10, 65K → 6528 tokens

**问题**: keep_ratio 只影响 drafter 输出，不影响 FP 计算！

FP 仍然计算完整的 65K tokens attention，只是最后丢弃了部分 tokens。

**真正的优化**: 使用更小的模型或更少的 layers

---

### 方案 D: 优化 ggml flash_attn_ext (长期)

需要修改 ggml backend 来优化 V100 上的 flash_attn_ext kernel。

---

## 结论

1. **GEMM block_score 优化无效** - drafter 不使用那个代码路径
2. **真正的瓶颈**: ggml flash_attn_ext (dense attention)
3. **V100 硬件限制**: SM70 不支持 BF16 WMMA 和 BSA
4. **推荐方案**: 使用 Ampere/Ada GPU 或等待 ggml 优化

