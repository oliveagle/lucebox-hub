#!/usr/bin/env python3
"""
收集 Qwen3.6 target hidden states 用于 draft 训练。

使用方法:
    python scripts/collect_draft_data.py \
        --model Qwen/Qwen3.6-27B \
        --output data/draft_training_qwen36.pt \
        --num-samples 100 \
        --max-length 2048
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

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


def collect_hidden_states(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int = 2048,
    max_new_tokens: int = 256,
) -> list[dict[str, Any]]:
    """收集指定层的 hidden states。

    Args:
        model: Target model (Qwen3.6-27B)
        tokenizer: Tokenizer
        prompts: List of prompts
        max_length: Maximum input length
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of samples with input_ids, target_ids, and hidden states
    """
    data = []

    for prompt in tqdm(prompts, desc="Collecting hidden states"):
        try:
            # Tokenize input
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
            # hidden_states is tuple of length (num_layers + 1)
            # Index 0 is embedding, so add offset of 1
            prompt_hidden = {
                f"layer_{i}": outputs.hidden_states[i + 1][:, -1:, :].cpu().clone()
                for i in TARGET_LAYERS
            }

            # Generate target tokens and capture generation hidden states
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )

            # Extract generated tokens only
            generated_ids = gen_ids.sequences[:, input_len:]

            # Build complete sample
            data.append({
                "input_ids": inputs["input_ids"].cpu(),
                "input_len": input_len,
                "target_ids": generated_ids.cpu(),
                "prompt_hidden": prompt_hidden,
                "prompt": prompt[:500],  # Truncate for storage
            })

        except Exception as e:
            logger.warning(f"Failed to process prompt: {e}")
            continue

    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect draft training data")
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
        default=100,
        help="Number of samples to collect",
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
        default="HuggingFaceH4/MATH-500",
        help="Dataset to use for prompts",
    )
    parser.add_argument(
        "--dataset-field",
        default="problem",
        help="Field in dataset to use as prompt",
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
    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info(f"Loading model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Target layers: {TARGET_LAYERS}")
    logger.info(f"Collecting {args.num_samples} samples")

    # Load model
    dtype = {"auto": "auto", "float16": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device,
        torch_dtype=dtype,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    try:
        ds = load_dataset(args.dataset, split="train")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Falling back to built-in prompts")
        ds = None
        prompts = [
            "Write a Python function to calculate fibonacci numbers.",
            "Explain the difference between TCP and UDP.",
            "What is the capital of France?",
            "Solve: 2 + 2 * 2 = ?",
            "Write a sorting algorithm in C++.",
        ] * (args.num_samples // 5 + 1)
        prompts = prompts[:args.num_samples]

    if ds is not None:
        num_samples = min(args.num_samples, len(ds))
        prompts = []
        for i in range(num_samples):
            try:
                prompt = ds[i][args.dataset_field]
                if isinstance(prompt, str):
                    prompts.append(prompt)
                elif isinstance(prompt, dict):
                    # Handle nested dicts
                    prompts.append(str(prompt))
                else:
                    prompts.append(str(prompt))
            except Exception as e:
                logger.warning(f"Failed to get sample {i}: {e}")
                continue

    # Collect data
    logger.info(f"Collecting {len(prompts)} samples...")
    start_time = time.time()

    data = collect_hidden_states(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    elapsed = time.time() - start_time
    logger.info(f"Collected {len(data)} samples in {elapsed:.1f}s ({elapsed/len(data):.1f}s per sample)")

    # Save data
    torch.save(data, args.output)
    logger.info(f"Saved {len(data)} samples to {args.output}")

    # Save metadata
    meta_path = output_path.with_suffix(".meta.json")
    metadata = {
        "num_samples": len(data),
        "target_layers": TARGET_LAYERS,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "model": args.model,
        "dataset": args.dataset,
        "collection_time": elapsed,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
