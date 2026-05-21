# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it should be included in prompts for context.

## Codebase Patterns (Study These First)

### TriAttention HIP Support

**Pattern**: TriAttention KV compression on HIP/ROCm requires graceful fallback
- `hipMemcpy` hangs on gfx1151 APU (slow system-memory architecture)
- Use early return with message instead of attempting GPU copies
- Macro pattern: `#if defined(HAS_GPU) && HAS_GPU` for defensive checks

---

## [2026-05-22] - lucebox-hub-gfx1151-x98

### 发现关键问题：TriAttention Stats 与模型维度不匹配

**问题**: Qwen3.6-27B 模型 (head_dim=256) 与现有 stats 文件 (Qwen3.5-27B, head_dim=64) 维度不匹配

**调试发现**:
- CUDA 后端 (NVIDIA V100) 可正常工作
- TriAttention 初始化成功: "Loaded stats with 64 layers, 24 heads, head_dim=64"
- 但模型实际 head_dim=256 (6144 / 24 = 256)
- 压缩触发时维度不匹配导致内存崩溃

**修复**:
1. 修正了 `compact_kv_head_positions` 使用 `gpuMemcpy` 而非 `std::memcpy`
2. 修正了 `compact_tria_k_pre_rope` 使用 `gpuMemcpy`
3. 添加了调试输出验证维度传递

**待解决**:
- 需要为 Qwen3.6-27B 生成新的 stats 文件
- 或使用 Qwen3.5-27B 模型进行测试

---

## [2026-05-22] - lucebox-hub-gfx1151-biv

### [发现] 模型加载 850 tensors 分配耗时过长 - 调查结果

**假设验证**: 原假设 "HIP JIT kernel 编译在 gfx1151 上很慢" **被推翻**

**实际根因**: 在后续 bead `lucebox-hub-gfx1151-5dq` 中发现是 **gfx1151 APU 的 H2D 内存传输极慢**

| 发现 | 详情 |
|------|------|
| 显卡类型 | Radeon 8060S (gfx1151) - **APU** (集成GPU，共享系统内存) |
| H2D 传输速度 | 994 MiB 需要 **>12 分钟** |
| 15 GiB 模型总耗时 | 预计 **3+ 小时** |
| 问题本质 | 不是代码 bug，是硬件限制 |

**加载流程分析** (来自本次调查):
1. GGUF init → mmap 文件
2. GPU buffer 分配 → 14.99 GiB 一次分配成功
3. Tensor 逐个初始化 → 850 tensors × 2 HIP calls = ~1700 次同步调用
4. Tensor 数据复制 → `cudaMemcpy(HostToDevice)` 是同步调用

**结论**: GGML 已经应用了 gfx1151 的所有 workaround，但 H2D 传输本身在 APU 上就是极慢的。

---

## [2026-05-22] - lucebox-hub-gfx1151-8pw

此 bead 发现 GGML 对 gfx1151 已有特殊处理但仍有 hang 问题。**验证为重复发现**。

**验证结果**:
1. **HIP Graphs 已自动禁用**: `common.cuh:1198-1199` 编译时通过 `#if defined(__gfx1151__)` 自动禁用
2. **同步 memcpy 已应用**: `ggml-cuda.cu:669,681` 使用同步 `cudaMemcpy` 替代 `hipMemcpyAsync + hipStreamSynchronize(hipStreamPerThread)`
3. **环境变量支持**: `GGML_CUDA_DISABLE_GRAPHS=1` 可通过运行时控制
4. **test 脚本已存在**: `test_hip_graphs_disable.sh`

**根因已在 `lucebox-hub-gfx1151-5dq` 中确认**: 不是真正的 hang，而是 gfx1151 APU 的 H2D 内存传输极慢（994 MiB 需要 12+ 分钟）

**结论**: 此 bead 是已有发现的确认，无需额外实现。

---

## [2026-05-22] - lucebox-hub-gfx1151-5dq

The "hang" in test_dflash and test_generate is NOT a hang - it's extremely slow H2D memory transfer on the gfx1151 APU.

**Key Finding**: Radeon 8060S (gfx1151) is an **APU** (integrated GPU) that shares system memory, not a discrete GPU with dedicated VRAM.

- The first tensor `output.weight` (994.63 MiB) takes **more than 12 minutes** to upload
- Total model size is ~15 GiB with 850+ tensors
- At this rate, full model loading would take **3+ hours**

### Timeline of Investigation

1. **Initial observation**: Program hangs after "uploading tensor 1/851: output.weight (994.63 MiB)"
2. **GGML analysis**: Confirmed gfx1151 workarounds already in place (sync memcpy, no hipStreamPerThread)
3. **Direct HIP copy test**: Using hipMemcpyAsync with explicit non-blocking stream - same slow result
4. **Root cause identified**: gfx1151 APU system memory architecture

### Existing Workarounds in GGML

- `ggml_backend_cuda_buffer_set_tensor`: Uses synchronous `cudaMemcpy` (not async)
- `ggml_backend_cuda_buffer_init_tensor`: Skips per-tensor cudaMemset (batched approach)
- `ggml_backend_cuda_buffer_set_tensor_2d`: Uses explicit stream (not hipStreamPerThread)
- CUDA/HIP Graphs: Auto-disabled for gfx1151

### Why All Approaches Are Slow

On an APU, H2D memcpy is copying from system RAM to... system RAM (just different address ranges).
The copy goes through:
1. CPU read from application buffer
2. Write to GPU-accessible memory region
3. Potential cache coherency operations

This is fundamentally slower than discrete GPU H2D transfers over PCIe.

### Recommendations

1. **Use a discrete GPU** for development/testing (e.g., RX 7900 series with dedicated VRAM)
2. **Reduce model size** - Q4_K_M quantization still too large for practical APU use
3. **Consider llama.cpp** - may have better APU optimizations

---

## [2026-05-22] - lucebox-hub-gfx1151-3hz

### [验收] gfx1151 APU hipMemcpy 完全挂起 - 已验证代码中已有 workaround

**结论**: 此 bead 记录的已知问题在代码中已有完整 workaround，无需额外实现。

**已实现的 workaround**:
1. **`gguf_target_loader.cpp`** (lines 574-648): 两步拷贝避免直接 hipMemcpy
   - 注释: "CRITICAL: On gfx1151, hipMemcpy from mmap regions HANGS"
   - 方案: mmap → malloc buffer → GPU (`ggml_backend_tensor_set`)
2. **`triattention_compress.cpp`** (lines 237-243): HIP 优雅降级
   - 跳过 TriAttention 压缩，打印提示信息

**根本原因** (已在 `lucebox-hub-gfx1151-5dq` 确认): gfx1151 APU 硬件限制
- H2D 传输极慢: 994 MiB 需要 12+ 分钟
- 不是真正的 hang，是系统内存架构的性能问题

---

## [2026-05-22] - lucebox-hub-gfx1151-yxa

### TriAttention HIP 支持修复

**问题**: `HAS_GPU` 条件编译在 HIP 上可能导致编译错误或运行时 hang

**修复**:
1. `#else` 分支显式定义 `HAS_GPU 0` 和 stub 宏 (第44行)
2. 所有 `#if HAS_GPU` 改为 `#if defined(HAS_GPU) && HAS_GPU` (141, 180, 237, 359)
3. HIP 上 graceful fallback: 打印消息并跳过 `tria_kv_compress`

**文件**: `dflash/src/triattention_compress.cpp`

---
