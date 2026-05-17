#!/usr/bin/env python3
"""
收集 Qwen3.6 target hidden states 用于 draft 训练。

支持多数据集收集: HumanEval (代码), MATH-500 (数学), ShareGPT (对话), 长文本

使用方法:
    python scripts/collect_draft_data.py \
        --model Qwen/Qwen3.6-27B \
        --output models/training_data/draft_training_qwen36.pt \
        --num-samples 10000

收集目标:
- >= 100K prompt-response 对
- 覆盖代码、数学、对话、长上下文场景
- 捕获 5 个目标层的 hidden states
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
from datasets import load_dataset
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))

# Target layer IDs for 27B model (64 layers)
# Use build_target_layer_ids(64, 5) = [1, 16, 31, 46, 60]
TARGET_LAYERS = [1, 16, 31, 46, 60]


@dataclass
class DatasetConfig:
    """数据集配置。"""
    name: str
    path: str
    field: str
    num_samples: int
    weight: float = 1.0


# 多数据集配置
DATASET_CONFIGS = [
    # 代码数据集
    DatasetConfig("humaneval", "openai/open-eval-extra", "problem", 500, 1.0),
    DatasetConfig("mbpp", "mbpp", "text", 500, 1.0),
    # 数学数据集
    DatasetConfig("math500", "HuggingFaceH4/MATH-500", "problem", 500, 1.0),
    DatasetConfig("gsm8k", "openai/gsm8k", "question", 1000, 1.0),
    # 对话数据集
    DatasetConfig("sharegpt", "anon8231489123/ShareGPT_V3_unfiltered", "conversations", 2000, 1.0),
    # 长上下文数据集
    DatasetConfig("longpqa", "汝묭/LongProcedureQA", "input", 200, 1.0),
    DatasetConfig("longalpaca", "yahma/alpaca-cleaned", "instruction", 500, 1.0),
]


def parse_conversations(convs: list) -> str:
    """解析 conversation 格式为单个字符串。"""
    if not convs:
        return ""
    parts = []
    for conv in convs:
        if isinstance(conv, dict):
            parts.append(conv.get("value", str(conv)))
        else:
            parts.append(str(conv))
    return "\n".join(parts)


def collect_from_dataset(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: DatasetConfig,
    max_length: int = 2048,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """从单个数据集收集 hidden states。"""
    data = []
    logger.info(f"Loading dataset: {config.name} from {config.path}")

    try:
        ds = load_dataset(config.path, split="train", trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Failed to load {config.name}: {e}, using fallback prompts")
        # Fallback prompts
        prompts = [
            "Write a Python function to calculate fibonacci numbers using recursion.",
            "Explain the difference between TCP and UDP protocols.",
            "What is the capital of France?",
            "Solve: 2 + 2 * 2 = ?",
            "Write a quicksort algorithm in C++.",
            "What is machine learning?",
            "Explain neural networks.",
            "What is the time complexity of binary search?",
        ] * (config.num_samples // 8 + 1)
        return collect_from_prompts(model, tokenizer, prompts[:config.num_samples], config.name, max_length, max_new_tokens)

    # Determine prompt extraction method
    num_samples = min(config.num_samples, len(ds))

    for i in tqdm(range(num_samples), desc=f"Collecting {config.name}"):
        try:
            sample = ds[i]
            if config.name == "sharegpt":
                # Handle conversation format
                raw = sample.get("conversations", sample)
                prompt = parse_conversations(raw if isinstance(raw, list) else [raw])
            else:
                prompt = sample.get(config.field, str(sample))

            if not isinstance(prompt, str) or len(prompt) < 10:
                continue

            # Truncate prompt to max_length
            prompt = prompt[:max_length * 4]  # Approximate char limit

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            # Capture prompt hidden states
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # Capture hidden states from target layers
            prompt_hidden = {
                f"layer_{idx}": outputs.hidden_states[idx + 1][:, -1:, :].cpu().clone()
                for idx in TARGET_LAYERS
            }

            # Generate target tokens
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Extract generated tokens
            generated_ids = gen_ids.sequences[:, input_len:]
            gen_len = generated_ids.shape[1]

            # Generate hidden states for each position (needed for training)
            gen_hidden = {}
            for step_idx in range(min(gen_len, max_new_tokens)):
                for layer_idx in TARGET_LAYERS:
                    key = f"layer_{layer_idx}"
                    if key not in gen_hidden:
                        gen_hidden[key] = []
                    # Get hidden at this step
                    step_hidden = gen_ids.hidden_states[step_idx + 1][layer_idx + 1]
                    gen_hidden[key].append(step_hidden[:, :1, :].cpu())

            # Concatenate generation hidden states
            for key in gen_hidden:
                gen_hidden[key] = torch.cat(gen_hidden[key], dim=1)

            data.append({
                "input_ids": inputs["input_ids"].cpu(),
                "input_len": input_len,
                "target_ids": generated_ids.cpu(),
                "prompt_hidden": prompt_hidden,
                "gen_hidden": gen_hidden,
                "source": config.name,
                "prompt": prompt[:500],
            })

        except Exception as e:
            logger.warning(f"Failed at sample {i}: {e}")
            continue

    logger.info(f"Collected {len(data)} samples from {config.name}")
    return data


def collect_from_prompts(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    source: str,
    max_length: int = 2048,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """从 prompts 列表收集数据。"""
    data = []

    for prompt in tqdm(prompts, desc=f"Collecting {source}"):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            prompt_hidden = {
                f"layer_{idx}": outputs.hidden_states[idx + 1][:, -1:, :].cpu().clone()
                for idx in TARGET_LAYERS
            }

            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            generated_ids = gen_ids.sequences[:, input_len:]

            gen_hidden = {}
            gen_len = generated_ids.shape[1]
            for step_idx in range(min(gen_len, max_new_tokens)):
                for layer_idx in TARGET_LAYERS:
                    key = f"layer_{layer_idx}"
                    if key not in gen_hidden:
                        gen_hidden[key] = []
                    step_hidden = gen_ids.hidden_states[step_idx + 1][layer_idx + 1]
                    gen_hidden[key].append(step_hidden[:, :1, :].cpu())

            for key in gen_hidden:
                gen_hidden[key] = torch.cat(gen_hidden[key], dim=1)

            data.append({
                "input_ids": inputs["input_ids"].cpu(),
                "input_len": input_len,
                "target_ids": generated_ids.cpu(),
                "prompt_hidden": prompt_hidden,
                "gen_hidden": gen_hidden,
                "source": source,
                "prompt": prompt[:500],
            })

        except Exception as e:
            logger.warning(f"Failed to process prompt: {e}")
            continue

    return data


def validate_data(data: list[dict[str, Any]]) -> dict[str, Any]:
    """验证数据质量。"""
    total = len(data)
    valid = 0
    nan_count = 0
    empty_count = 0

    for sample in data:
        if sample.get("target_ids") is None or len(sample.get("target_ids", [])) == 0:
            empty_count += 1
            continue

        # Check for NaN in hidden states
        has_nan = False
        for key, tensor in sample.get("prompt_hidden", {}).items():
            if torch.isnan(tensor).any():
                has_nan = True
                break

        if has_nan:
            nan_count += 1
            continue

        valid += 1

    return {
        "total": total,
        "valid": valid,
        "nan_count": nan_count,
        "empty_count": empty_count,
        "valid_rate": valid / total if total > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Qwen3.6 draft training data")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3.6-27B",
        help="Target model path or HuggingFace ID",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for collected data (.pt file)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Total number of samples to collect",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Maximum input sequence length",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate per sample",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        help="Dataset name to collect (can specify multiple, default: all)",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype (auto, float16, bfloat16)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--pilot-only",
        action="store_true",
        help="Run pilot collection with 100 samples per dataset",
    )
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine which datasets to collect
    if args.dataset:
        configs = [c for c in DATASET_CONFIGS if c.name in args.dataset]
    else:
        configs = DATASET_CONFIGS

    # Adjust samples for pilot mode
    if args.pilot_only:
        for cfg in configs:
            cfg.num_samples = min(cfg.num_samples, 100)
        logger.info("Pilot mode: collecting 100 samples per dataset")

    # Log configuration
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Target layers: {TARGET_LAYERS}")
    logger.info(f"Datasets: {[c.name for c in configs]}")
    total_samples = sum(c.num_samples for c in configs)
    logger.info(f"Collecting ~{total_samples} samples")

    # Load model
    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Collect data from each dataset
    all_data = []
    source_counts = {}

    for config in configs:
        dataset_data = collect_from_dataset(
            model=model,
            tokenizer=tokenizer,
            config=config,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
        )
        all_data.extend(dataset_data)
        source_counts[config.name] = len(dataset_data)

    # Validate data
    logger.info("Validating data quality...")
    validation = validate_data(all_data)
    logger.info(f"Validation: {validation['valid']}/{validation['total']} valid ({validation['valid_rate']:.1%})")

    if validation['nan_count'] > 0:
        logger.warning(f"Found {validation['nan_count']} samples with NaN values")
    if validation['empty_count'] > 0:
        logger.warning(f"Found {validation['empty_count']} empty samples")

    # Save data
    logger.info(f"Saving {len(all_data)} samples to {args.output}")
    torch.save(all_data, args.output)

    # Save metadata
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "num_samples": len(all_data),
        "target_layers": TARGET_LAYERS,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "model": args.model,
        "datasets": [c.name for c in configs],
        "source_counts": source_counts,
        "validation": validation,
        "collection_time": time.time(),
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")

    logger.info(f"Data collection complete: {len(all_data)} samples")
    logger.info("Next step: Run training with scripts/train_draft_qwen36.py")


if __name__ == "__main__":
    main()
