# DFlash + TriAttention 测试场景

> **创建日期**: 2026-05-19
> **关联 Epic**: lucebox-hub-gfx1151-dtri-vve (DFlash + TriAttention 纯 C++ 集成)
> **关联 Bead**: lucebox-hub-gfx1151-q32

## 测试场景定义

| 场景 | 模型 | 输入长度 | 输出长度 | 说明 |
|------|------|----------|----------|------|
| 短上下文对话 | Qwen3.6-27B-Q4_K_M | 4K | 256 | 常规对话，HumanEval/GSM8K 数据集 |
| 中上下文文档问答 | Qwen3.6-27B-Q4_K_M | 8K | 512 | 文档问答，Math500 数据集 |
| 长上下文检索 | Qwen3.6-27B-Q4_K_M | 32K | 512 | NIAH 检索测试，验证压缩精度 |
| 超长上下文极限 | Qwen3.6-27B-Q4_K_M | 64K | 1024 | 极限长上下文，验证内存压缩效果 |
| 131K 极限测试 | Qwen3.6-27B-Q4_K_M | 131K | 1024 | NIAH 131K 上下文验证 |

## 测试目标

### 短上下文对话 (4K → 256)
- **目标**: 验证 TriAttention 压缩在短上下文下的开销
- **验收标准**:
  - 压缩开销 < 5%
  - 精度与基线一致
  - 吞吐量无明显下降

### 中上下文文档问答 (8K → 512)
- **目标**: 验证中长上下文下的压缩效果
- **验收标准**:
  - KV 内存占用减少 2x+
  - 精度损失 < 1%
  - 吞吐量保持稳定

### 长上下文检索 (32K → 512)
- **目标**: 验证长上下文下的 NIAH 检索精度
- **验收标准**:
  - NIAH 检索准确率 100%
  - KV 内存占用减少 5x+
  - 端到端延迟改善

### 超长上下文极限 (64K → 1024)
- **目标**: 验证极限长上下文下的内存压缩效果
- **验收标准**:
  - 能够完成 64K 上下文推理
  - KV 内存占用显著降低
  - 精度可接受

### 131K 极限测试 (131K → 1024)
- **目标**: 验证模型最大上下文长度下的 TriAttention 表现
- **验收标准**:
  - 能够完成 131K 上下文推理
  - NIAH 检索准确率 ≥ 95%
  - KV 内存占用大幅降低

## 测试配置

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TRIATTN_ENABLED` | 1 | 启用 TriAttention |
| `TRIATTN_KV_BUDGET` | 2048 | 最大保留 token 数 |
| `TRIATTN_DIVIDE_LENGTH` | 128 | 压缩触发间隔 |
| `TRIATTN_WINDOW_SIZE` | 128 | 保留的最近 token |

### 编译选项

```bash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    -DDFLASH27B_TRIATTENTION=ON
```

### 测试命令

```bash
# 短上下文测试
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/dflash-draft-3.6-q8_0.gguf \
    prompt_4k.bin 256 output.bin

# 长上下文测试 (设置更大的 max_ctx)
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/dflash-draft-3.6-q8_0.gguf \
    prompt_32k.bin 512 output.bin \
    --max-ctx=65536
```

## 基准对比

### 对比维度

| 指标 | 无 TriAttention | 有 TriAttention | 目标 |
|------|-----------------|-----------------|------|
| KV 内存占用 | 基线 | 减少 2x+ | 5x+ (长上下文) |
| Decode 速度 | 基线 | 保持 | < 5% 开销 |
| 精度 | 基线 | 持平 | < 1% 差异 |
| 首次 Token 延迟 | 基线 | 持平 | 无显著增加 |

### 数据集

- **HumanEval**: 代码生成，10 样本
- **GSM8K**: 数学推理，10 样本
- **Math500**: 高级数学，10 样本
- **NIAH**: Needle-in-a-Haystack，多深度测试

## 测试执行计划

### 阶段 1: 短上下文验证
1. 编译 DFlash + TriAttention
2. 运行 HumanEval/GSM8K 测试
3. 记录吞吐量和精度

### 阶段 2: 长上下文验证
1. 运行 32K NIAH 测试
2. 验证检索准确率
3. 测量 KV 内存占用

### 阶段 3: 极限测试
1. 运行 64K/131K 极限测试
2. 验证内存压缩效果
3. 生成测试报告

## 交付物

- [ ] 测试结果数据 (JSON)
- [ ] 性能对比图表
- [ ] 精度验证报告
- [ ] 测试结论和建议
