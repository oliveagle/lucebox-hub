# V100 FlashPrefill 瓶颈分析与优化方案

**Date:** 2026-05-17
**GPU:** Tesla V100 (SM70, 72 SMs, 900 GB/s HBM2)

---

## 1. 当前瓶颈分析

### 1.1 FlashPrefill 四个阶段

| 阶段 | Kernel | 时间占比 (65K) | 说明 |
|------|--------|----------------|------|
| 1. mean_vector | `compute_mean_vector_kernel_f16` | 0.17 ms (0.6%) | ✅ 已优化 |
| 2. block_score | `compute_block_score_gemm_kernel_f16` | 0.31 ms (1.1%) | ✅ **已优化 (90×)** |
| 3. block_select | `block_select_kernel` | <1 ms (<3%) | ✅ 已优化 |
| 4. sparse_flash_forward | `sparse_flash_forward_kernel_f16` | ~28 ms (~97%) | ❌ **瓶颈** |

**结论**: 97% 的时间花在 `sparse_flash_forward` 上。

---

## 2. sparse_flash_forward_kernel_f16 分析

### 2.1 当前实现

```
配置: Q_TILE=64, K_TILE=64, BLOCK=128, D_HEAD=128
Grid: (q_tiles, batch * n_q_heads, 1) = (512, 16, 1) for 65K
Block: 64 threads (2 warps)
Shared Memory: ~43 KB (接近 Volta 48 KB 限制)
```

### 2.2 计算复杂度

对于每个 Q tile (64 tokens):
- 遍历选中的 K blocks (由 block_select 确定)
- 每个 K block 有 2 个 inner iterations (K_TILE=64)
- 每个 iteration 做:
  - Q @ K^T (WMMA, 32×8×16)
  - Softmax (在线更新 row max/logsum)
  - P @ V (WMMA, 32×8×16)

**关键问题**: `block_select` 选择的是 **dense blocks**，不是稀疏的！

当前实现选择的是 `top_k` blocks per query row，但 `sparse_flash_forward` 仍然计算 **full dense attention** over those selected blocks。

### 2.3 性能瓶颈

| 瓶颈 | 原因 | 影响 |
|------|------|------|
| **Global Memory Reads** | K/V 重复加载 (每个 Q tile 重复读取相同的 K/V blocks) | 高 |
| **Low Occupancy** | 64 threads/block, 仅 2 warps | 低 SM 利用率 |
| **Sequential Iterations** | 每个选中 block 串行处理 | 无法并行化 |
| **Shared Memory Pressure** | 43 KB 接近限制，难以扩大 tile | 限制了优化空间 |

---

## 3. 优化方案

### 方案 A: 减少 keep_ratio (立即可行 ⭐⭐⭐)

**目标**: 直接减少 sparse_flash_forward 的计算量

**原理**: `block_select` 根据 `alpha` 阈值选择 blocks。更低的 alpha = 更少的 blocks = 更少的 attention 计算。

**实现**:
```bash
# 当前: alpha=0.3, keep_ratio ~10%
# 优化: alpha=0.1, keep_ratio ~3%
echo "compress 100 8 32 5 /tmp/test_tokens.bin" | ./build/pflash_daemon model.gguf
                              # pool_kernel=5 (更低的 alpha)
```

**预期加速**: 2-3× (通过减少 selected blocks)

**权衡**: 可能略微降低 drafter 准确率，但 PFlash 对此有容错性

---

### 方案 B: Multi-CTA 并行化 (推荐 ⭐⭐)

**目标**: 增加 Occupancy，利用更多 SMs

**实现**:
```cpp
// 当前: 1 CTA per Q tile, 串行处理所有选中的 K blocks
// 优化: N CTA per Q tile, 每个处理一个 K block subset

// Grid: (q_tiles * num_ctas_per_qtile, batch * n_q_heads, 1)
// 每个 CTA 处理: selected_blocks / num_ctas_per_qtile 个 K blocks
// 最后 reduction 合并 O 结果
```

**预期加速**: 2-4× (取决于 num_ctas_per_qtile)

**挑战**:
- 需要在 CTA 间做 reduction (atomic add 或 tree reduction)
- 需要重新设计 shared memory layout

---

### 方案 C: 扩大 Tile Size (受限)

**目标**: 减少全局内存访问次数

**实现**:
```cpp
// 当前: Q_TILE=64, K_TILE=64
// 优化: Q_TILE=128, K_TILE=128 (需要更多 shared memory)

// 但 Volta shared memory 限制: 48 KB
// 当前: 64*128*2 + 64*128*2 + 64*64*2 + 128*4 = 43 KB
// 优化: 128*128*2 + 128*128*2 + 128*128*2 + 256*4 = 98 KB > 48 KB ❌
```

**结论**: 受限于 Volta 共享内存，无法简单扩大 tile。

---

### 方案 D: Kernel Fusion (长期)

**目标**: 减少中间结果的 global memory 写入

**当前流程**:
```
Q, K, V (global) → sparse_flash_forward → O (global)
```

**优化方案**:
```
融合 mean_vector + block_score + sparse_flash_forward
减少 Q, K, V 的重复加载
```

**挑战**:
- 复杂度高，需要重构整个 FlashPrefill pipeline
- 可能引入新的瓶颈

---

### 方案 E: 使用 CUDA Graph (工程优化)

**目标**: 减少 kernel launch overhead

**实现**:
```cpp
// 将整个 FlashPrefill pipeline 捕获为 CUDA Graph
// 减少 CPU-GPU 同步开销
cudaGraph_t graph;
cudaGraphCreate(&graph, 0);
// ... 添加所有 kernel nodes ...
cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
cudaGraphLaunch(graphExec, stream);
```

**预期加速**: 5-10% (主要减少 launch overhead)

---

### 方案 F: 真正的稀疏 Attention (架构级优化)

**目标**: 利用 block selection 的稀疏性

**问题**: 当前 `block_select` 选择 top-K blocks，但 `sparse_flash_forward` 仍然做 dense attention。

**优化方案**:
```cpp
// 对于每个 Q row, 只计算与其真正相关的 K tokens
// 而不是计算整个 selected block

// 当前: for each selected K block:
//         for each token in block:
//           compute attention
//
// 优化: 使用 block-level sparsity:
//       1. 识别每个 Q row 的真正 top-K tokens
//       2. 只计算那些 tokens 的 attention
```

**挑战**:
- 需要 irregular memory access patterns
- 可能降低 Tensor Core 利用率

---

## 4. 推荐的实施路径

### Phase 0: 立即验证 (今天 ⚡)

1. **测试更低 keep_ratio**
   ```bash
   # 当前: alpha=0.3 → keep_ratio ~10%
   # 测试: alpha=0.1 → keep_ratio ~3%

   # 生成 65K tokens
   python3 -c "
   import struct
   n = 65536
   with open('/tmp/test_tokens_65k.bin', 'wb') as f:
       f.write(struct.pack('<I', n))
       for i in range(n):
           f.write(struct.pack('<i', 1972))
   "

   # 使用更低 pool_kernel (影响 alpha)
   echo "compress 100 8 32 5 /tmp/test_tokens_65k.bin" | \
     DFLASH27B_V100_GEMM_SCORE=1 ./build/pflash_daemon \
     /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash/models/Qwen3-0.6B-BF16.gguf
   ```

2. **启用 profiling 确认瓶颈**
   ```bash
   DFLASH_FP_PROFILE=1 ./build/pflash_daemon model.gguf
   # 查看 stderr 输出的 stage timings
   ```

---

### Phase 1: Quick Wins (1-2 天)

3. **启用 CUDA Graph** (方案 E)
   - 低风险，工程改动小
   - 预期加速: 5-10%

4. **调优 block_select 参数**
   - 降低 `alpha`: 0.3 → 0.15 → 0.1
   - 减少 `window`: 2048 → 1024 → 512
   - 测试不同组合对准确率和性能的影响

---

### Phase 2: Kernel 优化 (1-2 周)

5. **Multi-CTA 并行化** (方案 B)
   - 每个高优先级，预期加速最大
   - 实现 CTA 间 reduction
   - 调优 num_ctas_per_qtile 参数

6. **Shared Memory 优化**
   - 减少冗余数据
   - 使用寄存器分片

---

### Phase 3: 架构优化 (长期)

7. **真正的稀疏 Attention** (方案 F)
   - 需要重新设计算法
   - 可能需要修改 block_select 逻辑

8. **考虑 Ampere/Ada GPU**
   - V100 的硬件限制是根本原因
   - RTX 3090/4090 有 3× Tensor Core 性能和 3.4× 共享内存

---

## 5. 立即可做的验证

```bash
# 1. 启用 profiling 确认瓶颈
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash
DFLASH_FP_PROFILE=1 ./build/pflash_daemon /path/to/model.gguf

# 2. 使用 nsight compute 分析 kernel
ncu --set full --export=report \
    ./build/pflash_daemon /path/to/model.gguf

# 3. 对比不同配置
for Q_TILE in 32 64 128; do
  # 需要修改代码支持不同 Q_TILE
  ./build/bench_flashprefill_e2e
done
```

---

## 6. 与 RTX 3090 对比

| 特性 | RTX 3090 (Ampere) | V100 (Volta) | 影响 |
|------|------------------|--------------|------|
| Tensor Core | 3rd gen (m16n16k16 BF16) | 1st gen (m32n8k16 F16) | 3× |
| Shared Memory | 164 KB | 48 KB | 3.4× |
| Memory BW | 936 GB/s | 900 GB/s | 1.04× |
| BSA Support | ✅ | ❌ | N/A |

**结论**: V100 的硬件限制是根本原因，Ampere 架构有显著优势。

---

## 7. 优化结果 ✅

### 原始性能基线

| Context | Time (s) | 瓶颈 |
|---------|----------|------|
| 65536 | 81.48 | ggml flash_attn_ext (77.11s, 94.8%) |

**根本原因**: Drafter 使用 `flash_prefill_forward_q8()` → `ggml_flash_attn_ext` (dense causal attention)，未使用优化的 WMMA kernels。

### 优化后性能 (2026-05-17)

通过混合 attention 方案（方向 C）+ GEMM block_score 优化：

| Context | Scalar (s) | GEMM (s) | Speedup | vs Baseline |
|---------|------------|----------|---------|-------------|
| 4096 | 0.653 | 0.648 | 1.01× | - |
| 16384 | 2.288 | 2.223 | 1.03× | - |
| 32768 | 5.248 | 4.824 | 1.09× | - |
| **65536** | **12.088** | **10.169** | **1.19×** | **8.0×** |

### 目标达成

- **目标**: 65K < 20s
- **结果**: 10.17s (GEMM) / 12.09s (Scalar)
- **加速**: 相比基线 81.48s，提升 **8.0×**

### 关键改进

1. **混合 Attention**: 使用 `ggml_flash_attn_sparse` + 注册 pFlash kernel
2. **GEMM block_score**: 长序列下提供 19% 加速
3. **Block sparsity**: keep_ratio=0.10，实际保留 10% tokens

### 已实施优化

- ✅ 方向 A: ggml WMMA kernel 集成（通过 flash_attn_sparse）
- ✅ 方向 B: DFlash F16 WMMA Kernel（部分，GEMM block_score）
- ✅ 方向 C: 混合方案（完整实现）

---

## 8. 最终建议

1. **✅ 已完成**: 混合 attention + GEMM block_score
2. **未来优化**: Multi-CTA 并行化（可进一步加速 2-4×）
3. **架构升级**: Ampere/Ada GPU 有 3× Tensor Core 性能
4. **替代方案**: 更低 keep_ratio (0.05 → 0.02) 减少 selected blocks 数量

