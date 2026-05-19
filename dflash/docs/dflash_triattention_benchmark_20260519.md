# DFlash + TriAttention 端到端 Benchmark 报告

**日期**: 2026-05-19
**GPU**: Tesla PG503-216 (GV100, 32GB HBM2)
**CUDA**: 12.2, Driver 535.288.01
**分支**: gfx1151

## 1. 编译验证

### CMake 配置

| 选项 | 值 | 说明 |
|------|-----|------|
| `DFLASH27B_TRIATTENTION` | ON | TriAttention KV 压缩 |
| `GGML_TRIATTENTION` | ON | ggml TriAttention 支持 |
| `DFLASH27B_GPU_BACKEND` | cuda | CUDA 后端 |

### 编译产物

| 文件 | 大小 | 说明 |
|------|------|------|
| `libdflash27b.a` | 1.0 MB | 静态库，含 TriAttention 符号 |
| `test_dflash` | 620.7 KB | DFlash 解码测试可执行文件 |
| `test_generate` | 106.0 KB | AR 基线生成测试可执行文件 |

### TriAttention 符号链接验证

```
# test_dflash 中 TriAttention 符号数量
nm test_dflash | grep -c tria → 10 个符号已链接
```

已链接的关键符号：
- `_ZN9dflash27b26init_triattention_from_envEv` (init_triattention_from_env)
- `_ZN9dflash27b17free_triattentionEv` (free_triattention)
- `_ZN9dflash27b17TriAttentionStateD1Ev` (TriAttentionState 析构)
- `tria_free`, `tria_load`, `tria_score_kv_head` (C 库函数)

### Stats 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `qwen3-1.7b.bin` | 448 KB | Qwen3-1.7B 注意力分数 |
| `qwen3-8b.bin` | 1.1 MB | Qwen3-8B 注意力分数 |
| `qwen3.5-27b.bin` | 768 KB | Qwen3.5-27B 注意力分数 |
| `qwen3_6_27b_stats.pt` | 1.6 MB | Qwen3.6-27B 注意力分数 (PyTorch 格式) |

## 2. 测试配置

### 模型

| 角色 | 路径 | 大小 | 量化 |
|------|------|------|------|
| Target | `models/Qwen3.6-27B-Q4_K_M.gguf` | ~16 GB | Q4_K_M |
| Draft | `models/draft/dflash-draft-3.6-q8_0.gguf` | ~1.7 GB | Q8_0 |

### 环境变量

TriAttention KV 压缩通过以下环境变量控制：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRIATTN_ENABLED` | - | 启用 TriAttention 压缩 (需要 stats 文件) |
| `TRIATTN_STATS_PATH` | `dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin` | Stats 文件路径 |
| `TRIATTN_KV_BUDGET` | 2048 | 最大保留 token 数 |
| `TRIATTN_DIVIDE_LENGTH` | 128 | 压缩间隔 (token 数) |
| `TRIATTN_WINDOW_SIZE` | 128 | 最近窗口 (始终保留) |

### Benchmark 命令

**无压缩基线**:
```bash
cd dflash/build
python3 scripts/bench_llm.py --bench HumanEval --budget 22
```

**启用 TriAttention 压缩**:
```bash
cd dflash/build
export TRIATTN_ENABLED=1
export TRIATTN_STATS_PATH=deps/llama.cpp/triattention/stats/qwen3.5-27b.bin
export TRIATTN_KV_BUDGET=2048
export TRIATTN_DIVIDE_LENGTH=128
export TRIATTN_WINDOW_SIZE=128
python3 scripts/bench_llm.py --bench HumanEval --budget 22
```

## 3. 显存预估

基于 Qwen3.6-27B-Q4_K_M 模型的分析：

| 组件 | 预估显存 (GB) | 说明 |
|------|---------------|------|
| Target 模型权重 | ~16.0 | Q4_K_M 量化后 |
| Draft 模型权重 | ~1.7 | Q8_0 量化后 |
| KV Cache (4K, 无压缩) | ~4.5 | Q4_0 K + Q8_0 V |
| KV Cache (4K, 压缩 50%) | ~2.3 | TriAttention 压缩后 |
| 激活 + 中间状态 | ~3.0 | forward pass 峰值 |
| **总计 (无压缩)** | **~25.2** | 4K 上下文 |
| **总计 (压缩 50%)** | **~23.0** | 4K 上下文, 50% KV 保留 |

**长上下文场景 (32K)**：

| 场景 | 预估 KV Cache (GB) | 总显存 (GB) |
|------|-------------------|-------------|
| 无压缩 | ~36.0 | **OOM** (超 32GB) |
| 压缩 50% | ~18.0 | ~39.0 (可能 OOM) |
| 压缩 25% | ~9.0 | ~30.0 (可行) |

## 4. 吞吐量预估

基于之前实验数据 (llama.cpp 基线 ~42.4 tokens/s, DFlash 在不同场景下 ~25-45 tokens/s)：

| 场景 | 无压缩 (tok/s) | 有压缩 (tok/s) | 压缩比 |
|------|---------------|---------------|--------|
| 4K → 256 | ~40 | ~38-40 | <5% 影响 |
| 8K → 512 | ~35 | ~32-35 | 5-10% 影响 |
| 32K → 512 | OOM | ~20-25 | 使原本不可行的场景可用 |

## 5. 输出一致性验证

TriAttention 压缩后的输出应与无压缩基线在短上下文中保持一致：

- **短上下文 (< 4K)**: 压缩不触发或影响极小，输出一致性 > 99%
- **中上下文 (4K-16K)**: 压缩触发后，输出质量取决于保留的 KV 预算
- **长上下文 (> 16K)**: 压缩使模型能在有限显存下运行，但可能影响需要远距离注意力的检索任务

## 6. 当前状态

- [x] CMake 编译选项 `DFLASH27B_TRIATTENTION` 已添加并配置
- [x] 编译成功: `libdflash27b.a`, `test_dflash`, `test_generate` 均已构建
- [x] TriAttention 符号已正确链接 (10 个符号验证通过)
- [x] Stats 文件可用 (4 个模型)
- [ ] 完整 benchmark 未运行 (GPU 显存不足，仅 1.4GB 空闲)
- [ ] 输出一致性对比未完成

## 7. 下一步

在 GPU 空闲时运行：

```bash
# 1. 基线测试 (无压缩)
cd dflash/build
python3 scripts/bench_llm.py --bench HumanEval GSM8K

# 2. 压缩测试
export TRIATTN_ENABLED=1
export TRIATTN_KV_BUDGET=2048
python3 scripts/bench_llm.py --bench HumanEval GSM8K

# 3. 不同预算对比
for BUDGET in 512 1024 2048 4096; do
  TRIATTN_KV_BUDGET=$BUDGET python3 scripts/bench_llm.py --bench HumanEval
done
```
