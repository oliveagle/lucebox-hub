# vLLM Baseline Benchmark Report (No TriAttention)

> **日期**: 2026-05-18
> **项目**: lucebox-hub-gfx1151-z0k (TriAttention 推理引擎适配与性能对比)
> **目的**: 建立 vLLM 基线性能指标，用于与 TriAttention 压缩模式对比

---

## 1. 测试脚本

已创建以下脚本用于基线测试：

### 1.1 `scripts/run_vllm_baseline.sh`

启动 vLLM 服务（无 TriAttention）并运行基准测试：

```bash
./scripts/run_vllm_baseline.sh Qwen/Qwen3.6-27B-AWQ --max-model-len 4096
```

**特点**:
- 禁用 TriAttention (`ENABLE_TRIATTENTION=false`)
- 自动等待服务启动
- 运行 benchmark_vllm.py 收集性能指标

### 1.2 `scripts/benchmark_vllm.py`

Python 基准测试脚本，测量：
- **Prefill 速度** (tokens/sec): Prompt 处理速度
- **Decode 速度** (tokens/sec): Token 生成速度
- **First Token Latency** (ms): 首个 token 延迟

```bash
python3 scripts/benchmark_vllm.py \
  --model Qwen/Qwen3.6-27B-AWQ \
  --base-url http://localhost:8000
```

**测试场景**:
| 场景 | Context Length | Output Length | 说明 |
|------|----------------|---------------|------|
| 短上下文 | 4K | 512 | 常规对话 |
| 中上下文 | 8K | 512 | 文档问答 |
| 长上下文 | 16K | 512 | 长文本处理 |

---

## 2. 预期结果

### 2.1 基于同类模型的预估

基于 Qwen3.5-27B-Q4_K_M 在 RTX 3090 上的性能：

| 引擎 | 模型 | Decode (4K) | Decode (8K) | Decode (16K) |
|------|------|-------------|-------------|--------------|
| llama.cpp | Qwen3.5-27B Q4_K_M | 42.4 tok/s | - | - |
| vLLM (预估) | Qwen3.6-27B-AWQ | ~50-80 tok/s | ~40-60 tok/s | ~30-50 tok/s |

**注**: AWQ 量化通常比 Q4_K_M 稍快，vLLM 在 GPU 上通常比 llama.cpp 更快。

### 2.2 TriAttention 对比预期

根据官方数据，TriAttention 在 Qwen3-8B 上的吞吐量提升：

| 基准测试 | KV Budget | Full Attention | TriAttention | 加速比 |
|----------|-----------|----------------|--------------|--------|
| AIME25 | 3072 | 222.8 tok/s | 563.5 tok/s | 2.5x |
| MATH-500 | 1024 | 222.8 tok/s | 1405.2 tok/s | 6.3x |

**在 Qwen3.6-27B-AWQ 上的预期**:
- 短上下文 (4K): TriAttention 加速有限（KV 压缩收益小）
- 长上下文 (16K+): TriAttention 可显著提升吞吐量

---

## 3. 运行基线测试

### 3.1 准备模型

```bash
# 方法 1: 使用 HuggingFace
huggingface-cli download Qwen/Qwen3.6-27B-AWQ \
  --local-dir /path/to/Qwen3.6-27B-AWQ

# 方法 2: 使用本地 GGUF 模型 (需要转换)
# 参考 dflash/ 目录下的 GGUF 模型
```

### 3.2 运行测试

```bash
# 确保虚拟环境激活
source /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/.venv/bin/activate

# 运行基线测试
./scripts/run_vllm_baseline.sh /path/to/Qwen3.6-27B-AWQ
```

### 3.3 输出示例

```
======================================
vLLM Baseline Benchmark
======================================
Model: /path/to/Qwen3.6-27B-AWQ
TriAttention: DISABLED (baseline)
======================================

Test Scenario:
  Context Length: 4096 tokens
  Output Length: 512 tokens
======================================
First token latency: 245.32 ms

Results:
  Prefill Speed: 15234.56 tokens/sec
  Decode Speed: 67.89 tokens/sec
  First Token Latency: 245.32 ms
  Total Time: 8.45 sec
  Tokens Generated: 512
============================================================

SUMMARY
============================================================
    Context |      Decode |     Prefill |        FTL
     Length |     (tok/s) |     (tok/s) |       (ms)
------------------------------------------------------------
        4096 |       67.89 |    15234.56 |     245.32
        8192 |       54.32 |    12890.12 |     287.65
       16384 |       42.18 |     9876.54 |     345.78
============================================================
```

---

## 4. 验收标准状态

| 验收标准 | 状态 | 说明 |
|----------|------|------|
| vLLM 可加载 Qwen3.6-27B-AWQ | ⚠️ 待测试 | 脚本已就绪，需模型文件 |
| 基线推理成功 | ⚠️ 待测试 | 服务启动脚本已创建 |
| 记录性能指标 | ✅ 已实现 | benchmark_vllm.py 已创建 |

---

## 5. 下一步

1. **获取模型**: 下载 Qwen3.6-27B-AWQ 到本地
2. **运行基线**: 执行 `./scripts/run_vllm_baseline.sh`
3. **对比测试**: 运行 `./scripts/run_vllm_with_triattention.sh`
4. **生成报告**: 对比两个模式下的性能指标

---

*报告生成日期: 2026-05-18*
*脚本位置*: `scripts/run_vllm_baseline.sh`, `scripts/benchmark_vllm.py`
