# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.2

### 完成状态: 已实现（待执行）

**实现内容：**
- `scripts/collect_draft_data.py` - 数据收集 pipeline（多数据集、hidden states 捕获、质量验证）
- `scripts/run_collect_draft_data.py` - 运行包装器（pilot 模式、依赖检查、数据合并）
- `scripts/train_draft_qwen36.py` - 完整训练 pipeline（DFlashDraftModel、训练循环、检查点保存）
- 参考 `deps/z-lab-dflash/dflash/model.py` 中的 `extract_context_feature` 函数

**文件变更：**
- 新增: `scripts/collect_draft_data.py` (444 行)
- 新增: `scripts/run_collect_draft_data.py` (316 行)
- 新增: `scripts/train_draft_qwen36.py` (581 行)

**目标层配置：**
```python
TARGET_LAYERS = [1, 16, 31, 46, 60]  # 64 层模型的 5 个代表性层
```

**数据集配置：**
```python
DATASET_CONFIGS = [
    DatasetConfig("humaneval", "openai/open-eval-extra", "problem", 500, 1.0),
    DatasetConfig("mbpp", "mbpp", "text", 500, 1.0),
    DatasetConfig("math500", "HuggingFaceH4/MATH-500", "problem", 500, 1.0),
    DatasetConfig("gsm8k", "openai/gsm8k", "question", 1000, 1.0),
    DatasetConfig("sharegpt", "anon8231489123/ShareGPT_V3_unfiltered", "conversations", 2000, 1.0),
    DatasetConfig("longpqa", "汝묭/LongProcedureQA", "input", 200, 1.0),
    DatasetConfig("longalpaca", "yahma/alpaca-cleaned", "instruction", 500, 1.0),
]
# 合计: 4700 samples/pilot, 全部约 100K+ samples
```

**执行要求：**
- GPU: V100 32GB 或 A100 40GB（需要 > 20GB 显存加载 27B 模型）
- 模型: HuggingFace 格式 Qwen3.6-27B（非 GGUF，需要能捕获 hidden states）
- 依赖: torch, transformers, datasets, loguru

**使用示例：**
```bash
# Pilot 测试
python scripts/run_collect_draft_data.py --pilot

# 完整收集
python scripts/run_collect_draft_data.py

# 指定数据集
python scripts/run_collect_draft_data.py --datasets humaneval math500

# 指定本地模型
python scripts/run_collect_draft_data.py --model-path /path/to/Qwen3.6-27B

# 合并已有数据
python scripts/run_collect_draft_data.py --merge file1.pt file2.pt --output-dir models/training_data
```

**Learnings:**
- DFlash draft 训练需要目标模型 hidden states，GGUF 格式不支持此操作
- 需要 HuggingFace transformers 格式的模型
- 数据收集使用 `output_hidden_states=True` 捕获中间层
- `extract_context_feature` 函数负责从 hidden_states 列表中提取指定层

---

