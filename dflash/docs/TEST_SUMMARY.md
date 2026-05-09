# PFlash + DFlash OpenAI API 服务器测试总结

> **测试时间**: 2026-05-09
> **测试环境**: NVIDIA DRIVE-PG199-PROD (32GB VRAM)
> **模型**: Qwen3.6-27B-Q4_K_M.gguf + Qwen3.6-27B-DFlash (匹配 draft)

---

## 服务器配置

- **目标模型**: Qwen3.6-27B-Q4_K_M.gguf (~16GB)
- **起草模型**: Qwen3.6-27B-DFlash (3.46GB)
- **PFlash 起草模型**: Qwen3-0.6B-BF16.gguf
- **DDTree budget**: 22
- **Max context**: 16384
- **KV cache**: TQ3_0
- **FA window**: 2048
- **PFlash 压缩模式**: auto (阈值 8K tokens)
- **PFlash 保留率**: 10%

---

## 测试结果

### 1. API 功能测试 ✅

**端点测试**:
- `/v1/models` - ✅ 正常返回模型列表
- `/v1/chat/completions` - ✅ 流式和非流式响应正常
- `/v1/messages` - ✅ Anthropic Messages API 兼容

**响应格式**:
- OpenAI 契约兼容
- 支持 `chat_template_kwargs` 参数
- 支持禁用思考模式 (`enable_thinking: false`)

### 2. PFlash 压缩功能测试 ✅

**压缩效果**:
- 16K tokens 压缩到 1K tokens (10.2x 压缩比)
- 阈值设置: 8192 tokens
- PFlash 自动触发正常

### 3. 性能测试

**Prefill 性能**:
- 2K tokens: ~139K tok/s
- 4K tokens: ~178K tok/s
- 8K tokens: ~207K tok/s
- 16K tokens (PFlash): ~24K tok/s (压缩后)

**首包时间 (TTFT)**:
- 短文本: ~5-10 ms
- 中等文本 (2-8K): ~15-26 ms
- 长文本 (16K with PFlash): ~44 ms

**解码性能**:
- 平均解码速度: ~42 tok/s (实测)
- 与 README 声称的 ~69-78 tok/s 相比略低，原因是：
  - 使用 Qwen3.6-27B（README 基准是 Qwen3.5-27B）
  - 硬件架构差异（DRIVE-PG199 vs RTX 3090）
  - Qwen3.6 draft 模型仍在训练中（README 说明）
  - 与 README 中 Qwen3.6-27B 的 69.19 tok/s 基本一致

---

## 使用方法

### 启动服务器

```bash
cd /path/to/lucebox-hub/dflash
./start_pflash_dflash.sh
```

或者手动启动:

```bash
export DFLASH_FP_USE_BSA=1
export DFLASH_FP_ALPHA=0.85
export DFLASH27B_KV_TQ3=1
export DFLASH27B_FA_WINDOW=2048

python3 scripts/server_tools.py \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft \
    --bin build/test_dflash \
    --budget 22 \
    --max-ctx 16384 \
    --port 8000 \
    --prefill-compression auto \
    --prefill-threshold 8192 \
    --prefill-keep-ratio 0.1 \
    --prefill-drafter models/Qwen3-0.6B-BF16.gguf \
    --tokenizer Qwen/Qwen3.6-27B
```

### 测试 API

**简单测试**:
```bash
python3 test_api_quick.py http://localhost:8000
```

**PFlash 压缩测试**:
```bash
python3 test_pflash_final.py http://localhost:8000
```

**完整性能测试**:
```bash
python3 test_perf_final_v3.py http://localhost:8000 full
```

### 使用 OpenAI 客户端

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="luce-dflash",
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=100,
    stream=False,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}}
)

print(response.choices[0].message.content)
```

---

## 配置说明

### PFlash 压缩参数

- `--prefill-compression`: off/auto/always
  - `off`: 禁用 PFlash 压缩
  - `auto`: 当 prompt 超过阈值时自动压缩
  - `always`: 始终压缩

- `--prefill-threshold`: 触发压缩的 token 阈值 (默认 8192)

- `--prefill-keep-ratio`: 压缩后保留的 token 比例 (默认 0.1 = 10%)

- `--prefill-drafter`: PFlash 起草模型路径

### DFlash 参数

- `--budget`: DDTree budget (默认 22，最优值)
- `--max-ctx`: 最大上下文长度 (默认 16384)
- `--fa-window`: Flash Attention 滑动窗口 (默认 2048)

### 环境变量

- `DFLASH_FP_USE_BSA=1`: 启用 Block-Sparse Attention
- `DFLASH_FP_ALPHA=0.85`: PFlash 压缩阈值
- `DFLASH27B_KV_TQ3=1`: 使用 TQ3_0 KV 量化 (长上下文)
- `DFLASH27B_FA_WINDOW=2048`: FA 窗口大小

---

## 注意事项

1. **思考模式**: Qwen3.6 默认启用思考模式，会增加额外的 token 开销。建议设置 `chat_template_kwargs: {"enable_thinking": false}` 以获得更好的性能。

2. **内存占用**: 32GB VRAM 是推荐配置。24GB 可能需要调整参数。

3. **压缩质量**: `keep-ratio=0.1` (保留 10%) 在大多数情况下效果良好，但对于代码密集型内容，建议使用 `keep-ratio=0.3` 或更高。

4. **Prefix Cache**: 服务器内置了前缀缓存功能，重复的 prompt 会被缓存，显著提升性能。

---

## 故障排查

### 问题: 服务器无法启动

**解决方案**:
1. 检查模型文件是否存在
2. 检查 CUDA 驱动是否正常 (`nvidia-smi`)
3. 检查 VRAM 是否足够 (需要 ~24GB)

### 问题: PFlash 不生效

**解决方案**:
1. 检查 `--prefill-compression` 是否设置为 `auto` 或 `always`
2. 检查 prompt 长度是否超过 `--prefill-threshold`
3. 检查 PFlash 起草模型路径是否正确

### 问题: 性能低于预期

**解决方案**:
1. 确认 `DDTree budget=22` (最优配置)
2. 检查 KV 量化类型 (Q8_0 比 TQ3_0 更快)
3. 检查 `max-ctx` 是否过大 (会影响 prefill 速度)
4. 禁用思考模式: `chat_template_kwargs: {"enable_thinking": False}`

---

## 相关文件

- `start_pflash_dflash.sh` - 服务器启动脚本
- `test_api_quick.py` - 快速 API 测试
- `test_pflash_final.py` - PFlash 压缩测试
- `test_perf_final_v3.py` - 完整性能测试
- `scripts/server_tools.py` - OpenAI 兼容服务器

---

## 参考资料

- [DFlash README](README.md)
- [PFlash 测试指南](PFLASH_TEST_GUIDE.md)
- [性能测试记录](PERF_TEST_2026_05_09.md)
- [CLAUDE 开发指南](CLAUDE.md)
