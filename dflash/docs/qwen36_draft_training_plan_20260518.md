# Qwen3.6 Draft 训练方案

**Date**: 2026-05-18
**Epic**: dflash-dflash-qwen36-acceptance-60-yzp
**GPU**: Tesla V100 32GB

---

## 1. 问题分析

### 当前状态

| 平台 | Target | Draft | AL | Accept% | Source |
|------|--------|-------|-------|---------|--------|
| RTX 3090 | Qwen3.5 | Qwen3.5 draft | 8.33 | ~65% | RESULTS.md |
| RTX 3090 | Qwen3.6 | Qwen3.6 draft | 5.05 | 32.3% | v100_performance_test |
| **V100** | Qwen3.6 | Qwen3.6 draft | 6.05 | 37.8% | v100_performance_test |

### 根本原因

1. **SWA (Sliding Window Attention)**: Qwen3.6 引入 SWA 层，模式为 `[true,true,true,true,false]`
2. **训练数据不匹配**: 当前 draft 是在 Qwen3.5 数据上训练的
3. **Hidden States 分布差异**: Qwen3.6 的 hidden states 与 Qwen3.5 不同

### 目标

- **Accept Rate**: 60%+ (当前 32-38%)
- **AL**: 8.0+ (当前 5-6)
- **V100 Decode**: 60+ tok/s (当前 38.81 tok/s)

---

## 2. DFlash Draft 架构

### 模型参数

```python
# Qwen3.6-27B Draft Model
HIDDEN = 5120           # 与 target 相同
N_LAYER = 5             # 5 层 draft
N_HEAD = 32
N_HEAD_KV = 8
HEAD_DIM = 128
INTERMEDIATE = 17408
BLOCK_SIZE = 16         # 每步预测 16 个 token
N_TARGET_LAYERS = 5     # 捕获 5 个 target 层特征
```

### 前向传播

```python
# Input:
# 1. noise_embedding = [last_token, MASK×15] token embeddings
# 2. target_hidden = concat([layer10, layer20, layer30, layer40, layer50])

# Output:
# 16 token logits (non-causal)

# Loss:
cross_entropy(
    draft_logits[:, 1:, :],      # 跳过位置 0 (last token 已知)
    target_tokens[:, 1:1+16]     # 接下来的 16 个 ground truth tokens
)
```

### Layer 选择

```python
# 27B model 有 64 层
target_layer_ids = [10, 20, 30, 40, 50]  # 或使用 build_target_layer_ids()

def build_target_layer_ids(num_target_layers, num_draft_layers):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1)))
            for i in range(num_draft_layers)]
```

---

## 3. 训练数据收集

### 数据源

使用 Qwen3.6-27B target 模型生成训练数据：

1. **代码数据**: HumanEval+, MBPP, LiveCodeBench
2. **数学数据**: MATH, GSM8K, Math500
3. **对话数据**: ShareGPT, OpenHermes
4. **长文本数据**: 需要 SWA context > 4K

### 数据格式

每个训练样本包含：

```python
{
    "input_ids": [101, 102, ..., 2048],      # 输入 tokens
    "target_ids": [2049, 2050, ..., 2560],  # 目标输出 tokens
    "hidden_states": {
        "layer_10": Tensor[seq_len, 5120],  # 捕获的 hidden states
        "layer_20": Tensor[seq_len, 5120],
        "layer_30": Tensor[seq_len, 5120],
        "layer_40": Tensor[seq_len, 5120],
        "layer_50": Tensor[seq_len, 5120],
    }
}
```

### 收集脚本

创建 `scripts/collect_draft_data.py`:

```python
"""
收集 Qwen3.6 target hidden states 用于 draft 训练。

    python scripts/collect_draft_data.py \
        --model Qwen/Qwen3.6-27B \
        --output data/draft_training_qwen36.pt \
        --num-samples 10000 \
        --max-length 2048
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Target layer IDs for 27B model (64 layers)
TARGET_LAYERS = [10, 20, 30, 40, 50]


def collect_hidden_states(model, tokenizer, prompts, max_length=2048):
    """收集指定层的 hidden states。"""
    data = []

    for prompt in tqdm(prompts):
        inputs = tokenizer(prompt, return_tensors="pt",
                          truncation=True, max_length=max_length)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False
            )

        # 捕获 5 个指定层的 hidden states
        hidden = {
            f"layer_{i}": outputs.hidden_states[i].cpu()
            for i in TARGET_LAYERS
        }

        # 生成 target tokens (用于训练 loss)
        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

        # 捕获生成过程中的 hidden states
        gen_hidden = {}
        for step_idx, step_output in enumerate(gen_ids.hidden_states):
            for layer_id in TARGET_LAYERS:
                key = f"layer_{layer_id}"
                if key not in gen_hidden:
                    gen_hidden[key] = []
                gen_hidden[key].append(step_output[layer_id].cpu())

        data.append({
            "input_ids": inputs["input_ids"].cpu(),
            "target_ids": gen_ids.sequences.cpu(),
            "prompt_hidden": hidden,
            "gen_hidden": gen_hidden,
        })

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 加载数据集
    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split="train")

    # 收集数据
    prompts = [ds[i]["problem"] for i in range(min(args.num_samples, len(ds)))]
    data = collect_hidden_states(model, tokenizer, prompts, args.max_length)

    # 保存
    torch.save(data, args.output)
    print(f"Saved {len(data)} samples to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 4. Draft 模型训练

### 训练脚本

创建 `scripts/train_draft_qwen36.py`:

```python
"""
训练 Qwen3.6 DFlash Draft 模型。

    python scripts/train_draft_qwen36.py \
        --data data/draft_training_qwen36.pt \
        --output models/draft/dflash-draft-3.6-trained.pt \
        --epochs 10 \
        --batch-size 8 \
        --lr 1e-4
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 导入 DFlash 模型
import sys
sys.path.append("deps/z-lab-dflash")
from dflash.model import DFlashDraftModel, build_target_layer_ids
from transformers import Qwen3Config, Qwen3ForCausalLM


class DraftDataset(Dataset):
    """Draft 训练数据集。"""

    def __init__(self, data_path, block_size=16):
        self.data = torch.load(data_path)
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Input: [last_token, MASK×16]
        input_ids = sample["input_ids"]
        last_token = input_ids[:, -1:]  # 最后一个 token

        # Target: 接下来的 16 个 tokens
        target_ids = sample["target_ids"]
        target_block = target_ids[:, :self.block_size]

        # Hidden states: concat 5 个层
        prompt_hidden = sample["prompt_hidden"]
        hidden = torch.cat([
            prompt_hidden[f"layer_{i}"][:, -1:, :]
            for i in [10, 20, 30, 40, 50]
        ], dim=-1)

        return {
            "last_token": last_token,
            "target_block": target_block,
            "target_hidden": hidden,
        }


def train_epoch(model, dataloader, optimizer, device):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader):
        last_token = batch["last_token"].squeeze(1).to(device)
        target_block = batch["target_block"].squeeze(1).to(device)
        target_hidden = batch["target_hidden"].squeeze(1).to(device)

        # 前向传播
        noise_embedding = model.model.embed_tokens(last_token)
        logits = model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        # 计算 loss (跳过第一个位置，因为是已知的 last_token)
        logits = logits[:, 1:, :]  # [batch, 15, vocab]
        target = target_block[:, 1:16]  # [batch, 15]

        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target.reshape(-1),
            ignore_index=-100
        )

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda")

    # 创建 config
    config = Qwen3Config(
        hidden_size=5120,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=17408,
        block_size=args.block_size,
        num_target_layers=64,  # Qwen3.6-27B 的层数
        dflash_config={
            "target_layer_ids": [10, 20, 30, 40, 50],
            "mask_token_id": 151644,  # Qwen tokenizer
        }
    )

    # 创建模型
    print("Creating DFlash draft model...")
    model = DFlashDraftModel(config).to(device)

    # 加载数据
    print(f"Loading data from {args.data}")
    dataset = DraftDataset(args.data, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # 训练
    for epoch in range(args.epochs):
        loss = train_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}: loss={loss:.4f}")

    # 保存
    torch.save(model.state_dict(), args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()
```

---

## 5. 转换为 GGUF

训练完成后，使用现有的转换脚本：

```bash
# 1. 保存为 safetensors 格式
model.save_pretrained("models/draft/dflash-draft-3.6-trained")

# 2. 转换为 GGUF (需要实现转换脚本)
python scripts/convert_dflash_to_gguf.py \
    --model models/draft/dflash-draft-3.6-trained \
    --output models/draft/dflash-draft-3.6-trained-q8_0.gguf \
    --quant q8_0
```

---

## 6. 验收测试

训练完成后，运行验收测试：

```bash
# 1. 基本功能测试
./build/test_dflash \
    models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/dflash-draft-3.6-trained-q8_0.gguf \
    /tmp/test_prompt.bin \
    256 \
    /tmp/test_output.bin \
    --fast-rollback \
    --ddtree \
    --ddtree-budget=22

# 2. HumanEval benchmark (10 samples)
python3 scripts/bench_llm.py --bench HumanEval --budget 22

# 3. 验收标准
# - Accept Rate >= 60%
# - AL >= 8.0
# - V100 decode speed >= 60 tok/s
```

---

## 7. 资源需求

### 计算资源

| 阶段 | GPU | 内存 | 时间 |
|------|-----|------|------|
| 数据收集 | V100 32GB | 32GB | ~24h (10K samples) |
| 模型训练 | V100 32GB | 32GB | ~48h (10 epochs) |
| 验证测试 | V100 32GB | 32GB | ~2h |

### 存储需求

- 训练数据: ~100GB (10K samples × 10MB)
- 模型 checkpoint: ~10GB (BF16) + 2GB (Q8_0 GGUF)

---

## 8. 依赖与风险

### 依赖

1. ✅ **V100 GPU**: 可用
2. ✅ **Qwen3.6-27B Target**: 已有 `models/Qwen3.6-27B-Q4_K_M.gguf`
3. ❌ **训练数据**: 需要从 target 收集
4. ❌ **官方训练脚本**: z-lab 尚未开源

### 风险

1. **数据收集时间**: 10K samples 需要 ~24h
2. **训练时间**: 10 epochs 需要 ~48h
3. **结果不确定**: 训练可能无法达到 60% 接受率
4. **V100 内存限制**: 32GB 可能不够大 batch size

### 缓解措施

1. **从少量数据开始**: 先用 1K samples 验证流程
2. **使用梯度累积**: 模拟更大 batch size
3. **混合精度训练**: FP16/BF16 减少内存
4. **监控验证集**: 防止过拟合

---

## 9. 时间线

| 阶段 | 时间 | 产出 |
|------|------|------|
| Phase 1: 数据收集 (1K samples) | 4h | pilot data |
| Phase 2: 训练 pilot 模型 | 6h | pilot draft |
| Phase 3: 验证 pilot 质量 | 2h | accept rate |
| Phase 4: 全量数据收集 (10K) | 24h | training data |
| Phase 5: 全量训练 | 48h | final draft |
| Phase 6: 验收测试 | 4h | benchmark results |
| **总计** | **~88h** | **production draft** |

---

## 10. 替代方案

如果 DIY 训练不可行，替代方案：

1. **等待 z-lab 官方训练脚本**: 预计 1-3 个月
2. **使用 EAGLE3**: 另一种 speculative decoding 方法
3. **切换回 Qwen3.5**: 已有 65% 接受率

---

## 11. 下一步行动

### 立即可做

1. ✅ 创建 `scripts/collect_draft_data.py` 数据收集脚本
2. ✅ 创建 `scripts/train_draft_qwen36.py` 训练脚本
3. ⏳ 运行 pilot 数据收集 (100 samples)
4. ⏳ 训练 pilot 模型
5. ⏳ 验证 accept rate 提升

### 需要决策

1. **是否投入 ~88h 训练时间**: 用户确认
2. **优先级**: 是否有其他更高优先级任务
3. **资源分配**: V100 是否可用用于训练

---

**Status**: 🟡 等待用户确认是否继续 DIY 训练方案
**Last Updated**: 2026-05-18
