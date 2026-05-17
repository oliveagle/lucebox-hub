# V100 FlashPrefill 优化 - 最终报告

**日期**: 2026-05-17
**Epic**: dflash-v100-flashprefill-optimization-v3i
**GPU**: Tesla PG503-216 (V100), SM70, 72 SMs, 32GB HBM2
**模型**: Qwen3-0.6B-BF16.gguf

---

## 1. 执行摘要

### 目标

将 65K tokens 的 FlashPrefill 时间从 **81.48s 降到 < 20s**（5× 加速）。

### 结果

| 指标 | 基线 | 优化后 | 提升 |
|------|------|--------|------|
| 65K Time | 81.48s | 10.17s | **8.0×** |
| 65K Throughput | ~805 tok/s | 6446 tok/s | **8.0×** |
| 目标达成 | - | < 20s | ✅ **超出 50%** |

### 关键成果

- ✅ 65K tokens 在 10.17s 内完成（GEMM kernel）
- ✅ 实现混合 attention 架构（block-select + ggml flash_attn_sparse）
- ✅ GEMM block_score 提供 19% 加速（长序列）
- ✅ 压缩比稳定在 10%（keep_ratio=0.10）

---

## 2. 优化方向

### 方向 A: 直接调用 ggml WMMA Kernel ✅

**实施**: 集成 ggml WMMA kernel 到 `flashprefill_q8.cpp`

**方法**:
- 使用 `ggml_flash_attn_sparse` 注册机制
- 通过 `pflash_ggml_adapter.cpp` 路由到 DFlash WMMA kernels
- 条件编译支持 Volta F16 和 Ampere+ BF16

**状态**: ✅ 完成

**文件变更**:
- `src/flashprefill_q8.cpp` - 修改为使用 `ggml_flash_attn_sparse`
- `src/pflash_ggml_adapter.cpp` - 添加命名空间和条件编译
- `CMakeLists.txt` - 在 sm_70 构建中添加 adapter

### 方向 B: 修复 DFlash F16 WMMA Kernel ⚠️

**问题**: `sparse_flash_forward_f16` kernel 在 S >= 16384 时崩溃

**调查结果**:
- block_select kernel 功能正常（已验证）
- 崩溃在 WMMA attention kernel 的内存访问
- 根因是复杂的 shared memory layout 和 WMMA 操作

**状态**: ⚠️ 部分完成
- ✅ GEMM block_score kernel 可用
- ❌ sparse_flash_forward 仍有问题（通过方向 C 绕过）

### 方向 C: 混合方案 Block-Select + GGML FA ✅

**实施**: 结合 DFlash block-select 和 ggml flash_attn_sparse

**方法**:
- block-select 选择稀疏 K blocks
- ggml flash_attn_sparse 计算密集 attention
- 通过 alpha 参数控制 sparsity

**状态**: ✅ 完整实现

**性能**: 这是最终生效的优化方案

---

## 3. 性能数据

### E2E 完整 Pipeline 性能

**配置**: keep_ratio=0.10, lookahead=8, chunk=32, pool=13

#### Scalar Kernel (默认)

| Context | Time (s) | Kept | Ratio | tok/s |
|---------|----------|------|-------|-------|
| 4096    | 0.653    | 1024 | 0.2500 | 6273 |
| 16384   | 2.288    | 1632 | 0.0996 | 7160 |
| 32768   | 5.248    | 3264 | 0.0996 | 6241 |
| 65536   | 12.088   | 6528 | 0.0996 | 5424 |

#### GEMM Kernel (DFLASH27B_V100_GEMM_SCORE=1)

| Context | Time (s) | Kept | Ratio | tok/s | Speedup |
|---------|----------|------|-------|-------|--------|
| 4096    | 0.648    | 1024 | 0.2500 | 6318 | 1.01× |
| 16384   | 2.223    | 1632 | 0.0996 | 7371 | 1.03× |
| 32768   | 4.824    | 3264 | 0.0996 | 6790 | 1.09× |
| **65536** | **10.169** | 6528 | 0.0996 | **6446** | **1.19×** |

### 与基线对比

| 配置 | 65K Time | vs Baseline |
|------|----------|-------------|
| 基线 (ggml flash_attn_ext) | 81.48s | 1.0× |
| 优化后 (Scalar) | 12.09s | **6.7×** |
| 优化后 (GEMM) | 10.17s | **8.0×** |

### GEMM 加速效果

- 4K: 几乎无加速 (1.01×)
- 16K: 3% 加速
- 32K: 9% 加速
- 65K: **19% 加速**

**原因**: 长序列下 block_score 占比增加，GEMM 优势更明显

---

## 4. 技术细节

### 数据流

```
Q/K/V (Q8) → ggml_cast → F32/F16
           → ggml_permute → [D,S,H]
           → ggml_cont (确保连续)
           → flash_attn_sparse (alpha < 1.0)
           → pFlash adapter
           → DFlash WMMA kernels
           → Output
```

### 关键发现

1. **ggml_cont 必要性**: `ggml_permute` 创建 view，需要 `ggml_cont` 确保数据连续
2. **条件类型转换**: 避免不必要的中间 buffer 分配
3. **命名空间**: `dflash27b::flashprefill` 包裹 adapter 避免链接冲突

### 编译配置

```bash
# V100 (SM70) 启用 GEMM block_score
DFLASH27B_V100_GEMM_SCORE=1

# 运行测试
echo "compress 100 8 32 13 /tmp/test_65k.bin" | \
  ./build/pflash_daemon ./models/Qwen3-0.6B-BF16.gguf
```

---

## 5. 代码模式与经验

### 模式: ggml_flash_attn_sparse 注册机制

- **文件**: `src/pflash_ggml_adapter.cpp`
- **用途**: 绕过 ggml op dispatch，直接调用自定义 kernel
- **优势**: 集成 DFlash kernels 到 ggml pipeline

### 模式: E2E vs Micro-benchmark

- **Micro-benchmark**: 测量单个 kernel 时间
- **E2E**: 测量完整 pipeline (prefill + score + compress)
- **注意**: tail-score 时间包含多个阶段，不能直接对比

### Gotcha: pflash_daemon 输入格式

- 需要 **binary token 文件**，不是文本
- 格式: `u32 count (LE) + count × int32 token IDs`
- 使用 `struct.pack('<I', n) + struct.pack('<i', token_id) * n` 生成

---

## 6. 未完成与未来工作

### 未完成项

- ❌ `sparse_flash_forward_f16` kernel 大序列崩溃（方向 B）
- 原因: WMMA kernel 共享内存布局复杂
- 绕过: 使用方向 C 混合方案

### 未来优化方向

1. **Multi-CTA 并行化**
   - 每个 Q tile 使用多个 CTA
   - 预期加速: 2-4×

2. **更低的 keep_ratio**
   - 0.10 → 0.05 → 0.02
   - 减少计算量，可能略微降低准确率

3. **CUDA Graph**
   - 减少 kernel launch overhead
   - 预期加速: 5-10%

4. **架构升级**
   - Ampere/Ada GPU 有 3× Tensor Core 性能
   - 支持 BSA (Block-Sparse Attention)

---

## 7. 复现指南

### 环境要求

- GPU: Tesla V100 (SM70) 或兼容
- CUDA 11.x+
- CMake 3.20+
- gcc 11+

### 编译

```bash
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub/dflash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 运行测试

```bash
# 生成测试数据
python3 -c "
import struct
n = 65536
with open('/tmp/test_65k.bin', 'wb') as f:
    f.write(struct.pack('<I', n))
    for i in range(n):
        f.write(struct.pack('<i', 1972))
"

# Scalar kernel
echo "compress 100 8 32 13 /tmp/test_65k.bin" | \
  ./build/pflash_daemon ./models/Qwen3-0.6B-BF16.gguf

# GEMM kernel
DFLASH27B_V100_GEMM_SCORE=1 ./build/pflash_daemon ./models/Qwen3-0.6B-BF16.gguf
```

---

## 8. 结论

V100 FlashPrefill 优化项目成功达成目标：

1. ✅ **65K < 20s**: 实测 10.17s（GEMM）/ 12.09s（Scalar）
2. ✅ **8.0× 加速**: 相比基线 81.48s
3. ✅ **混合 attention**: block-select + ggml flash_attn_sparse
4. ✅ **GEMM block_score**: 长序列下 19% 加速

主要贡献是实现了混合 attention 架构，通过 `ggml_flash_attn_sparse` 注册机制将 DFlash block-select 与 ggml flash attention infrastructure 集成。
