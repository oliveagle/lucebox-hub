# PRD: TriAttention KV 压缩集成

## 1. 概述与目标

### 项目背景

**lucebox-hub-gfx1151** 是一个本地 LLM 推理服务器，使用 C++/CUDA/HIP 构建的 DFlash 推测解码引擎，支持多种模型和硬件。**TriAttention** 是一个基于三角频率域评分的高效 KV cache 压缩系统，可将 KV 内存减少 10.7 倍，同时保持精度。

### 整合目标

将 TriAttention 集成到 lucebox-hub-gfx1151 项目中，实现：
1. **KV 内存压缩**: 在长推理任务中减少 10.7x KV 内存占用
2. **吞吐量提升**: 在 AIME25 基准测试中实现 2.5x 吞吐量提升
3. **兼容现有架构**: 不破坏现有的 DFlash 推测解码流水线

### 整合模式选择

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| llama.cpp GGUF | TriAttention 已在 llama.cpp 中支持 (HIP/ROCm) | DFlash 推测解码后端 |
| vLLM | TriAttention 的官方集成 | 高吞吐服务部署 |
| SGLang | TriAttention 的实验性支持 | SGLang 用户 |

---

## 2. 当前架构分析

### lucebox-hub-gfx1151 架构

```
┌─────────────────────────────────────────────────────────────┐
│                     lucebox-hub-gfx1151                      │
├─────────────────────────────────────────────────────────────┤
│  dflash/ (C++/CUDA/HIP)                                     │
│  ├── src/           # 核心推理引擎                           │
│  │   ├── flashprefill.cpp/h   # 推测预填充                   │
│  │   ├── delta_net_chunked.cpp/h  # SSM 状态管理             │
│  │   └── flashprefill_kernels.cu/hip  # CUDA/HIP 内核       │
│  ├── deps/llama.cpp/  # GGUF 量化支持                        │
│  └── deps/Block-Sparse-Attention/  # BSA 内核               │
│                                                              │
│  triattention/ (新增子模块)                                 │
│  ├── triattention/vllm/  # vLLM 集成                        │
│  ├── triattention/sglang/  # SGLang 集成                    │
│  ├── triattention/core/  # 核心压缩算法                      │
│  └── scripts/  # 标定工具                                   │
└─────────────────────────────────────────────────────────────┘
```

### DFlash 推测解码流水线

```
Token Stream → DFlash Draft → Target Model → DDTree Verify → Output
                  ↓
            Hidden States
                  ↓
            KV Cache (压缩目标)
```

### TriAttention 压缩流程

```
KV Cache → 三角频率评分 → Top-K 选择 → 内存压缩
            ↑
     Precomputed Stats (.pt)
```

---

## 3. 功能需求

### FR-1: TriAttention 子模块集成

**描述**: 将 triattention 仓库作为 git submodule 集成到项目中

**验收标准**:
- [ ] `submodules/triattention/` 目录存在
- [ ] `git submodule status` 显示 triattention 已注册
- [ ] 可以正常 `git submodule update --init`

### FR-2: llama.cpp + TriAttention 支持

**描述**: 在 DFlash 的 llama.cpp 后端中启用 TriAttention KV 压缩

**实现方式**:
- 利用已有的 triattention-ggml 社区实现
- 编译时启用 TriAttention 支持
- 通过环境变量配置 KV 预算

**验收标准**:
- [ ] llama.cpp 编译支持 TriAttention
- [ ] 可以设置 `TRIATTN_RUNTIME_KV_BUDGET=2048` 等参数
- [ ] 与 DFlash draft 模型兼容

### FR-3: vLLM + TriAttention 支持

**描述**: 在 Python vLLM 集成路径中启用 TriAttention

**实现方式**:
- 使用 `triattention.vllm.runtime.integration_monkeypatch`
- 配置环境变量和统计文件路径

**验收标准**:
- [ ] vLLM 服务启动时显示 `[TriAttention] Runtime (V2) plugin activated`
- [ ] 可以加载 `triattention/vllm/stats/` 中的预计算统计
- [ ] 长上下文推理时 KV 内存占用显著降低

### FR-4: 统计文件生成 (Calibration)

**描述**: 为支持的模型生成预计算频率统计文件

**实现方式**:
- 使用 `scripts/calibrate.py` 脚本
- 为 Qwen3-8B、DeepSeek-R1 等模型生成 .pt 统计文件

**验收标准**:
- [ ] 可以为自定义模型生成统计文件
- [ ] 统计文件格式与 vLLM 加载器兼容

### FR-5: 压缩效果验证

**描述**: 验证 TriAttention 压缩不会影响推理精度

**测试场景**:
- AIME24/AIME25 数学推理基准
- MATH-500 性能测试
- 长上下文 NIAH (Needle-in-a-Haystack) 检索

**验收标准**:
- [ ] 压缩后精度与 Full Attention 持平 (差异 < 1%)
- [ ] KV 内存占用减少 5x 以上
- [ ] 端到端延迟改善

---

## 4. 非功能需求

### NFR-1: 性能要求

| 指标 | 目标 | 说明 |
|------|------|------|
| 吞吐量提升 | ≥ 2x | vs Full Attention on AIME25 |
| KV 内存压缩 | ≥ 5x | 在 2048 budget 下 |
| 延迟开销 | < 5% | 压缩操作本身 |

### NFR-2: 兼容性要求

- 兼容 AMD gfx1151 (Strix Halo) HIP 后端
- 兼容 NVIDIA CUDA 后端
- 不破坏现有 DFlash 推测解码功能

### NFR-3: 可配置性

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `TRIATTN_RUNTIME_KV_BUDGET` | 2048 | 最大保留 token 数 |
| `TRIATTN_RUNTIME_DIVIDE_LENGTH` | 128 | 压缩触发间隔 |
| `TRIATTN_RUNTIME_WINDOW_SIZE` | 128 | 保留的最近 token |
| `TRIATTN_RUNTIME_SPARSE_STATS_PATH` | - | 统计文件路径 |
| `ENABLE_TRIATTENTION` | true | 主开关 |

---

## 5. 用户故事

### US-1: 长推理任务加速

**作为** 开发者/用户
**我想要** 在处理长推理任务时减少 KV 内存占用
**以便** 在有限的 GPU 显存中运行更长的上下文

**验收标准**:
- 32K+ 上下文长度下，显存占用减少 5x
- 推理精度保持不变

### US-2: 集成到现有流水线

**作为** 开发者
**我想要** 将 TriAttention 无缝集成到现有的 DFlash 流水线中
**以便** 不需要修改现有的服务代码

**验收标准**:
- 只需设置环境变量即可启用
- 不影响现有的 llama.cpp 或 DFlash 接口

### US-3: 自定义模型支持

**作为** 高级用户
**我想要** 为自定义模型生成 TriAttention 统计文件
**以便** 享受 KV 压缩带来的性能提升

**验收标准**:
- 提供 calibration 脚本
- 生成的统计文件可被 vLLM 加载

---

## 6. 技术约束与决策

### TC-1: vLLM 多进程架构

vLLM 使用 scheduler/worker 分离的多进程架构，TriAttention 的压缩需要在 worker 进程中执行。这与 DFlash 的单进程 C++ 架构不同。

**决策**: 提供两种集成路径:
1. **DFlash 路径**: llama.cpp + TriAttention (C++/ggml)
2. **vLLM 路径**: 独立使用 vLLM + TriAttention

### TC-2: 标定数据依赖

TriAttention 需要预计算的 Q/K 频率统计文件。这些文件需要针对每个模型架构生成。

**决策**: 提供默认统计文件 + 自定义生成工具

### TC-3: 与 DFlash 的交互

TriAttention 压缩 KV cache，而 DFlash 推测解码依赖 KV cache。需要确保两者兼容。

**决策**: 先在 decode 阶段应用 TriAttention，保留 prefill 阶段的完整 KV

---

## 7. 里程碑

| 阶段 | 交付物 | 完成标准 |
|------|--------|----------|
| M1: 子模块集成 | `submodules/triattention/` | git submodule 正常工作 |
| M2: llama.cpp 支持 | 编译 + 基础功能 | 可以启动带 TriAttention 的服务 |
| M3: vLLM 支持 | Python 集成 | vLLM 启动日志显示 plugin activated |
| M4: 标定工具 | `scripts/calibrate.py` | 可生成 .pt 统计文件 |
| M5: 验证测试 | 基准测试报告 | AIME25 精度 + 性能数据 |

---

## 8. 相关文档

- [TriAttention README](submodules/triattention/README.md)
- [TriAttention vLLM Integration](submodules/triattention/docs/vllm.md)
- [TriAttention SGLang Integration](submodules/triattention/docs/sglang.md)
- [TriAttention Calibration Guide](submodules/triattention/docs/calibration.md)
- [lucebox-hub dflash README](dflash/README.md)