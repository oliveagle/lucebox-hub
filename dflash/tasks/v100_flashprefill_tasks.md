# V100 FlashPrefill 优化任务列表

**项目**: V100 FlashPrefill Optimization
**截止日期**: 2026-05-24

---

## Critical Priority

### [V100-001] 定位 block_select bug 并修复
**状态**: `todo`
**估时**: 2d
**标签**: #bug #volta #flashprefill

分析 `flashprefill_f16.cu` 的 `block_select_kernel`，定位导致崩溃的内存访问问题。测试 S=320, 350, 4096 等边界情况。

**关键文件**: `src/flashprefill_f16.cu:602-668`

**检查点**:
- [ ] 添加边界检查 `if (n >= N) n = N - 1`
- [ ] 添加 `cudaMemset(dIdx, -1, idxb)` 初始化
- [ ] 验证 grid 配置正确性
- [ ] 测试 S=320, 350, 4096, 65536

---

### [V100-004] 添加 block_select 边界检查和初始化
**状态**: `todo`
**估时**: 1d
**标签**: #bugfix #volta
**依赖**: V100-001

在 `block_select_kernel` 中添加完整的边界检查，确保 `n < N`。添加 `cudaMemset` 确保内存初始化。

---

### [V100-005] 验证 DFLASH_FP_USE_VOLTA_FP=1 不崩溃
**状态**: `todo`
**估时**: 1d
**标签**: #testing #volta
**依赖**: V100-004

测试 S=320..65536 范围内所有序列长度，确保强制启用 Volta FP 不崩溃。

---

## High Priority

### [V100-002] 研究直接调用 ggml WMMA kernel 的可行性
**状态**: `todo`
**估时**: 3d
**标签**: #integration #ggml #flashprefill

在 `flashprefill_q8.cpp` 中直接调用 `ggml_cuda_flash_attn_ext_wmma_f16`，绕过 op dispatch。

**参考**: `deps/llama.cpp/ggml/src/ggml-cuda/fattn-wmma-f16.cu:558`

---

### [V100-003] 集成 ggml WMMA kernel 并验证正确性
**状态**: `todo`
**估时**: 2d
**标签**: #integration #flashprefill
**依赖**: V100-002

将方向 A 的 kernel 调用集成到 `flashprefill_q8.cpp`，绕过 ggml op dispatch。

---

### [V100-008] 端到端测试: pflash_daemon
**状态**: `todo`
**估时**: 1d
**标签**: #testing #e2e
**依赖**: V100-005, V100-007

运行 `pflash_daemon` 测试，验证 4K, 16K, 32K, 65K context 场景下的正确性和输出质量。

---

### [V100-009] 性能基准测试
**状态**: `todo`
**估时**: 1d
**标签**: #benchmark #performance
**依赖**: V100-008

测量所有优化方向的性能，生成 benchmark 报告。

**目标**:
- 16K: < 3s (当前 6.96s)
- 32K: < 8s (当前 22.56s)
- 65K: < 20s (当前 81.48s)

---

## Medium Priority

### [V100-006] 实现混合方案: block-select + ggml FA
**状态**: `todo`
**估时**: 5d
**标签**: #feature #hybrid #flashprefill
**依赖**: V100-003, V100-005

结合 DFlash 的 block-select 算法和 ggml 的 flash_attn_ext:
1. 收集选中的 K block indices
2. 提取子矩阵 K_selected, V_selected
3. 调用 ggml_flash_attn_ext(Q, K_selected, V_selected)
4. 结果写回 O

---

### [V100-007] 优化混合方案的数据移动
**状态**: `todo`
**估时**: 2d
**标签**: #optimization #performance
**依赖**: V100-006

最小化 K/V 数据复制，使用 inplace 操作，减少中间 buffer 分配。

---

### [V100-010] 写性能报告
**状态**: `todo`
**估时**: 0.5d
**标签**: #docs
**依赖**: V100-009

生成最终的 benchmark 报告，总结三个方向的优化效果。

---

## 任务依赖关系

```
V100-001 (定位bug)
    ↓
V100-004 (添加边界检查)
    ↓
V100-005 (验证不崩溃)

V100-002 (研究ggml)
    ↓
V100-003 (集成)
    ↓
V100-006 (混合方案) → V100-007 (优化)
    ↓
V100-008 (e2e测试) → V100-009 (基准) → V100-010 (报告)
```

---

## 验收标准

三个方向**至少一个**达到:

1. **方向 A (ggml)**: 直接调用成功，性能可测量
2. **方向 B (修复)**: `DFLASH_FP_USE_VOLTA_FP=1` 不崩溃
3. **方向 C (混合)**: 端到端通过，性能优于纯 GGML

**最低标准**: 65K context 性能提升 ≥ 30% (81.48s → < 57s)

---

## 快速开始

```bash
# 1. 定位 bug
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash

# 2. 测试当前状态
echo "compress 100 8 32 13 /tmp/test_tokens_4096.bin" | \
  ./build/pflash_daemon models/Qwen3-0.6B-BF16.gguf

# 3. 测试 Volta FP (当前崩溃)
DFLASH_FP_USE_VOLTA_FP=1 echo "compress 100 8 32 13 /tmp/test_tokens_4096.bin" | \
  ./build/pflash_daemon models/Qwen3-0.6B-BF16.gguf

# 4. 运行 benchmark
./build/bench_flashprefill_e2e
```

---

*Created: 2026-05-17*
*Source: docs/prd_v100_flashprefill_optimization_20260517.md*