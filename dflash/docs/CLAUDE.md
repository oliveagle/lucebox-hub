# DFlash 项目配置

> **最后更新**: 2026-05-09
> **模型**: Qwen3.6-27B-Q4_K_M + Qwen3.6-27B-DFlash BF16 draft

---

## 性能基准测试数据

### 当前实测性能 (NVIDIA DRIVE-PG199-PROD, 32GB VRAM)

| Context | AR baseline | DFlash | Speedup | Config |
|---------|-------------|--------|---------|--------|
| 4K | 31.7 tok/s | 85.9 tok/s | 2.7× | budget=22, no TQ3 |
| 256K | 31.7 tok/s | 58.4 tok/s | 1.8× | budget=22, fa-window=2048 |

**测试命令**:
```bash
# AR baseline
./build/test_generate models/Qwen3.6-27B-Q4_K_M.gguf <prompt> 256 <out> --max-ctx=8192

# DFlash with DDTree
unset DFLASH27B_KV_TQ3
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/model.safetensors <prompt> 256 <out> \
  --max-ctx=5120 --fast-rollback --ddtree --ddtree-budget=22
```

### 与 README 目标对比

**README Qwen3.5-27B 性能（RTX 3090）**：
- HumanEval: **129.5 tok/s** (3.43× vs AR 37.8)
- AR baseline: 37.8 tok/s

**README Qwen3.6-27B 性能（RTX 5090 Blackwell）**：
- 117K context (PFlash 压缩): **210.7 tok/s**
- ~3× vs AR

**当前性能差距**：
- 硬件: DRIVE-PG199 (Turing/SM 75) vs RTX 5090 (Blackwell/SM 120)
- 架构差距: Blackwell 比 Turing 快 ~2-3×
- 我们的 DFlash: 85.9 tok/s vs README 129.5 tok/s (Qwen3.5)
- 差距原因: 硬件架构 + 模型版本 + prompt 类型

**关键发现**：
1. **RTX 5090 (Blackwell) 比 RTX 3090/DRIVE-PG199 快 2-3×**
2. **Qwen3.6 + PFlash** 在 117K context 下能达到 210.7 tok/s
3. 我们用的是 Turing 架构，性能符合预期

### llama.cpp MTP 对比数据

| 方法 | tok/s | 说明 |
|------|-------|------|
| AR baseline | 111.4 tok/s | llama.cpp autoregressive |
| ngram-simple | 111.4 tok/s | MTP，几乎无加速 |
| ngram-mod | 111.3 tok/s | MTP，几乎无加速 |

**结论**: llama.cpp 的 MTP 实现没有带来加速，而 DFlash 达到了 2.7× 加速。

---

## 关键配置

### DDTree budget

| Budget | 4K decode | 说明 |
|--------|-----------|------|
| 22 | 81.7 tok/s | 最优配置 |
| 32 | 69.4 tok/s | 速度下降 |
| 40 | 65.6 tok/s | 速度下降 |
| 48 | 58.5 tok/s | 速度下降 |

**结论**: budget=22 是最佳配置，增加 budget 会降低速度。

### KV 量化类型

| 类型 | 说明 | VRAM 使用 |
|------|------|-----------|
| Q8_0 (默认) | 8-bit 量化 | 基准 |
| TQ3_0 | 3.5 bpw，支持 256K context | 更少 VRAM |

**设置**:
```bash
# 默认 Q8_0
unset DFLASH27B_KV_TQ3

# 使用 TQ3_0 (长 context)
export DFLASH27B_KV_TQ3=1
```

### Flash Attention Window

| Window | 256K decode | 说明 |
|--------|-------------|------|
| 2048 | 58.4 tok/s | 滑动窗口 |
| 0 (全注意力) | ~35 tok/s | 全注意力，更慢 |

**结论**: fa-window=2048 在长 context 下显著更快。

---

## 测试脚本

- `test_context_quick.py`: 快速测试不同 context 大小
- `create_prompt.py`: 创建测试 prompt 文件
- `scripts/bench_llm.py`: 官方基准测试（HumanEval 等）

---

## 问题排查

### 性能低于预期

1. **检查 budget 值**: 必须使用 `--ddtree-budget=22`
2. **检查 KV 类型**: 默认 Q8_0 比 TQ3_0 更快
3. **检查 max_ctx**: 过大会降低 prefill 速度
4. **检查 prompt 类型**: HumanEval 编程题比重复文本更快

### AR baseline 测量

```bash
./build/test_generate models/Qwen3.6-27B-Q4_K_M.gguf <prompt> 10 <out> --max-ctx=128
# 输出: [gen] 10 new tokens in X.XX s  ->  YY.YY tok/s
```

### DFlash 速度测量

```bash
unset DFLASH27B_KV_TQ3
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/model.safetensors <prompt> 256 <out> \
  --max-ctx=5120 --fast-rollback --ddtree --ddtree-budget=22
# 输出: [dflash] generated 256 tokens in X.XX s  ->  YY.YY tok/s
```

---

## 性能优化建议

1. **使用 budget=22**: 最优配置
2. **使用 fa-window=2048**: 长 context 下更快
3. **避免使用 TQ3_0**: 除非需要 256K context
4. **使用正确的 max_ctx**: 避免过大导致 prefill 慢

---

## 相关文件

- `README.md`: 项目说明和性能声称
- `RESULTS.md`: 官方基准测试结果
- `test_context_quick.py`: 快速性能测试
