# V100 FlashPrefill 优化 PRD

**Date:** 2026-05-17
**Status:** Draft
**Assignee:** ralph-tui

---

## 1. 背景与问题

### 1.1 当前瓶颈

| 组件 | 时间 (65K) | 占比 |
|------|-----------|------|
| A_compute | 0.77s | 0.9% |
| **FlashPrefill** | **77.11s** | **94.8%** |
| tail_score | 0.73s | 0.9% |
| **Total** | **81.48s** | **100%** |

### 1.2 根因分析

1. **ggml flash_attn_ext 使用 DENSE attention** - O(S²) 复杂度
2. **V100 (SM70) 不支持 BSA** - Block-Sparse Attention 需要 SM80+
3. **DFlash 的 F16 WMMA kernel 有 bug** - block_select 导致崩溃

### 1.3 代码路径

```
drafter → forward_qwen3_drafter_model()
         → flash_prefill_forward_q8()  // 当前路径
         → ggml_flash_attn_ext()
         → ggml_cuda_flash_attn_ext_wmma_f16()  // V100 使用这个
```

**问题**: 虽然 ggml 自己的 `fattn-wmma-f16.cu` 有完整的 Volta WMMA 实现，但它是 DENSE attention，无法利用 block sparsity。

---

## 2. 目标

在 V100 (SM70) 上实现高性能 FlashPrefill，目标是:

| Context | 当前时间 | 目标时间 | 加速比 |
|---------|----------|----------|--------|
| 16K | 6.96s | < 2s | 3.5× |
| 32K | 22.56s | < 5s | 4.5× |
| 65K | 81.48s | < 15s | 5.4× |

---

## 3. 三个优化方向

### 方向 A: 直接调用 ggml 的 WMMA Kernel

**原理**: ggml 的 `fattn-wmma-f16.cu` 已经有完整的 Volta F16 WMMA FlashAttention 实现，可以直接集成。

**优势**:
- 利用已有的优化 kernel
- 兼容性好，不需要修改 ggml 本身
- 可以在 dflash 层做调用

**实施**:

```cpp
// 在 flashprefill_q8.cpp 中
// 1. 直接实例化并调用 fattn kernel
// 2. 绕过 ggml 的 op dispatch，直接调用 cuda kernel

// 参考: deps/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu
```

**参考代码**:

```cpp
// deps/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu:558
void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // 这个函数可以直接调用
    // 需要构造正确的 tensor 参数
}
```

### 方向 B: 修复 DFlash 的 F16 WMMA Kernel

**原理**: DFlash 的 `flash_prefill_forward_f16` 有完整的 block-select + sparse attention 实现，只是 block_select kernel 有 bug。

**Bug 定位**:

分析 `flashprefill_f16.cu` 的 block_select_kernel:

```cpp
// line 602-668: block_select_kernel
// 可能的 bug:
1. 边界检查不完整 (line 617: m >= M, 但 N 没有检查)
2. 共享内存访问越界
3. Warp ballot 操作在某些情况下返回错误 mask
```

**调试步骤**:

```cpp
// 1. 添加边界检查
if (n >= N) n = N - 1;  // 防止越界

// 2. 添加 memset 确保内存初始化
cudaMemset(dIdx, -1, idxb);

// 3. 验证 grid/block 配置
// 当前: grid=(B, M, H), block=32
// 检查: M * H * B 是否超过可用 SMs
```

**测试矩阵**:

| Sequence Length | Bug 触发 | 预期行为 |
|----------------|----------|----------|
| S=320 | Yes | Crash |
| S=350 | Yes | Crash |
| S=4096 | TBD | Test |
| S=65536 | TBD | Test |

### 方向 C: 混合方案 (Block-Select + GGML FA)

**原理**: 结合 DFlash 的 block-select 算法和 ggml 的优化 kernel。

**算法**:

```
1. mean_vector: 计算每个 K block 的 mean (已有)
2. block_score: 计算 Q blocks 与 K blocks 的相似度 (已有, 可用 GEMM)
3. block_select: 选择 top-K blocks (已有, 需要修复)
4. sparse_attention: 对选中的 blocks 做 FA
   - 方案 C1: 使用现有的 sparse_flash_forward (F16 WMMA)
   - 方案 C2: 使用 ggml flash_attn_ext 对选中 blocks 做 dense attention
```

**方案 C2 实现**:

```cpp
// 1. 收集选中的 K block indices
// 2. 提取子矩阵 K_selected, V_selected
// 3. 调用 ggml_flash_attn_ext(Q, K_selected, V_selected)
// 4. 结果写回 O
```

---

## 4. 技术要求

### 4.1 硬件约束

| 参数 | V100 值 |
|------|---------|
| Compute Capability | SM70 |
| Tensor Cores | 1st gen (m16n8k16 F16) |
| Shared Memory | 48 KB |
| Memory BW | 900 GB/s HBM2 |
| SMs | 72 |

### 4.2 Kernel 配置要求

| Kernel | Threads | SM Usage | Shared Memory |
|--------|---------|----------|---------------|
| mean_vector | 128 | per head | 0 |
| block_score (GEMM) | 512 | multi-CTA | ~16 KB |
| block_select | 32 | 1 warp | 0 |
| sparse_flash_forward | 64 | 2 warps | ~43 KB |
| ggml WMMA FA | 128 | variable | ~32 KB |

### 4.3 数据布局

```cpp
// 当前布局: [batch, seq, heads, dim]
// V100 WMMA 优化需要:
// - 16-byte aligned addresses
// - strides 是 16 的倍数
// - head_dim = 128 (标准)
```

---

## 5. 实施计划

### Phase 1: 调试与验证 (1-2 天)

#### Task 1.1: 定位 block_select bug

```
1. 添加详细的边界检查
2. 添加 cuda-memcheck 验证
3. 测试 S=320, 350, 4096 等边界情况
4. 验证 grid 配置正确性
```

**验收标准**: block_select 在 S=320..65536 范围内不崩溃

#### Task 1.2: 对比 GGML kernel vs DFlash kernel

```
1. 编写对比测试程序
2. 测试 16K, 32K, 65K context
3. 记录两种实现的性能差异
```

**验收标准**: 获得 baseline 性能数据

### Phase 2: 方向 A 实现 (2-3 天)

#### Task 2.1: 直接调用 ggml WMMA kernel

```
1. 在 flashprefill_q8.cpp 中添加 ggml WMMA 调用
2. 绕过 ggml op dispatch，直接调用 cuda kernel
3. 验证正确性和性能
```

**验收标准**: 集成成功，性能可测量

#### Task 2.2: 性能调优

```
1. 调整 tile size
2. 优化 shared memory 使用
3. 测试不同 batch size
```

### Phase 3: 方向 B 修复 (2-3 天)

#### Task 3.1: 修复 block_select

```
1. 逐一排查可能的问题点
2. 添加边界检查和初始化
3. 回归测试确保不引入新问题
```

**验收标准**: block_select 在所有支持的 S 值上正确工作

#### Task 3.2: 集成并测试

```
1. 启用 flash_prefill_forward_f16
2. 测试 DFLASH_FP_USE_VOLTA_FP=1 不崩溃
3. 测量性能提升
```

### Phase 4: 方向 C 实现 (3-5 天)

#### Task 4.1: 实现混合 attention

```
1. 提取选中 blocks 的子矩阵
2. 调用 ggml flash_attn_ext
3. 结果写回主矩阵
```

#### Task 4.2: 优化数据移动

```
1. 最小化 K/V 数据复制
2. 使用 inplace 操作
3. 减少中间 buffer 分配
```

**验收标准**: 混合方案工作正常，性能优于纯 GGML

### Phase 5: 集成与测试 (2-3 天)

#### Task 5.1: 端到端测试

```
1. 运行 pflash_daemon 测试
2. 测试 4K, 16K, 32K, 65K context
3. 验证正确性 (输出 token 质量)
```

#### Task 5.2: 性能基准

```
1. 记录所有场景的 TTFT
2. 对比优化前后性能
3. 确认达到目标
```

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| block_select bug 难以定位 | 时间延迟 | 优先尝试方向 A |
| GGML kernel 集成复杂度高 | 可能引入新 bug | 充分单元测试 |
| V100 硬件限制 | 无法达到 Ampere 性能 | 设定合理的性能目标 |

---

## 7. 验收标准

### 7.1 功能验收

- [ ] block_select 在 S=320..65536 范围内稳定运行
- [ ] DFLASH_FP_USE_VOLTA_FP=1 不崩溃
- [ ] 端到端测试通过 (pflash_daemon)
- [ ] 输出 token 质量无下降

### 7.2 性能验收

| Context | 当前 | 目标 | 必须达成 |
|--------|------|------|----------|
| 16K | 6.96s | < 3s | 否 |
| 32K | 22.56s | < 8s | 否 |
| 65K | 81.48s | < 20s | 否 |

**注**: "必须达成" 指无论如何都要达到的最低标准

### 7.3 代码质量

- [ ] 无 compiler warnings
- [ ] 通过 cuda-memcheck
- [ ] 单元测试覆盖率 > 80%
- [ ] 文档更新

---

## 8. 参考资料

### 8.1 代码文件

```
关键文件:
- src/flashprefill_f16.cu          # F16 WMMA 实现 (有 bug)
- src/flashprefill_f16_gemm.cu     # GEMM block_score (正常)
- src/flashprefill_q8.cpp         # 当前调用的 GGML 封装
- src/flashprefill.cpp            # dispatch 层

GGML kernel:
- deps/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu    # Volta WMMA FA
- deps/llama.cpp/ggml/src/ggml-cuda/fattn.cu             # kernel dispatch
- deps/llama.cpp/ggml/src/ggml-cuda/fattn-common.cuh     # 共享定义
```

### 8.2 测试工具

```bash
# E2E benchmark
./build/bench_flashprefill_e2e

# 全流程测试
echo "compress 100 8 32 13 /tmp/test_tokens_65k.bin" | \
  ./build/pflash_daemon models/Qwen3-0.6B-BF16.gguf

# 启用 profiling
DFLASH_FP_PROFILE=1 ./build/pflash_daemon ...

# 强制 Volta FP (当前会崩溃)
DFLASH_FP_USE_VOLTA_FP=1 ./build/pflash_daemon ...
```

### 8.3 Benchmark 结果

| Sequence | Scalar block_score | GEMM block_score | Speedup |
|----------|-------------------|-----------------|---------|
| 4096 | 0.30 ms | 0.04 ms | 7.7× |
| 8192 | 0.80 ms | 0.04 ms | 20.8× |
| 16384 | 2.49 ms | 0.06 ms | 38.4× |
| 32768 | 8.27 ms | 0.12 ms | 68.4× |
| 65536 | 28.24 ms | 0.31 ms | 89.7× |

---

## 9. 成功标准

三个方向中**至少一个**达到以下标准:

1. **block_select 修复成功**: `DFLASH_FP_USE_VOLTA_FP=1` 不崩溃
2. **性能提升 ≥ 30%**: 65K context 从 81.48s 降到 < 57s
3. **混合方案工作**: block-select + ggml FA 端到端通过

---

## 10. 下一步行动

1. **立即开始**: Task 1.1 - 定位 block_select bug
2. **并行**: Task 2.1 - 研究直接调用 ggml kernel 的可行性
3. **待定**: Task 4.1 - 混合方案实现 (取决于前两个方向的进展)

---

*PRD Created: 2026-05-17*
*Last Updated: 2026-05-17*