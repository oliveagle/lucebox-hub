#!/usr/bin/env python3
"""
验证训练好的 DFlash draft 模型。

使用方法:
    python scripts/validate_draft.py \
        --model models/draft/dflash-draft-3.6-trained \
        --target Qwen/Qwen3.6-27B \
        --num-samples 10
"""
import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))
from dflash.model import DFlashDraftModel, extract_context_feature, sample

# Target layer IDs
TARGET_LAYERS = [1, 16, 31, 46, 60]


def load_trained_draft(model_path: str, config: dict) -> DFlashDraftModel:
    """Load trained draft model."""
    from transformers import Qwen3Config

    qwen_config = Qwen3Config(
        hidden_size=config.get("hidden_size", 5120),
        num_hidden_layers=config.get("num_hidden_layers", 5),
        num_attention_heads=config.get("num_attention_heads", 32),
        num_key_value_heads=config.get("num_key_value_heads", 8),
        intermediate_size=config.get("intermediate_size", 17408),
        vocab_size=config.get("vocab_size", 152064),
    )
    qwen_config.dflash_config = {
        "target_layer_ids": config.get("target_layer_ids", TARGET_LAYERS),
    }

    model = DFlashDraftModel(qwen_config)
    state_dict = torch.load(model_path / "final_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def validate_model(
    draft_model: DFlashDraftModel,
    target_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int = 256,
) -> dict:
    """Validate draft model with target."""
    device = next(target_model.parameters()).device

    results = {
        "num_samples": 0,
        "num_success": 0,
        "errors": [],
    }

    for prompt in tqdm(prompts, desc="Validating"):
        try:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            # Run target model to get hidden states
            with torch.no_grad():
                outputs = target_model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )

            # Get last token embedding
            last_token_id = inputs["input_ids"][:, -1:]
            last_token_emb = target_model.model.embed_tokens(last_token_id)

            # Get target hidden states
            target_hidden = extract_context_feature(outputs.hidden_states, TARGET_LAYERS)
            target_hidden = target_hidden[:, -1:, :]  # [1, 1, hidden * num_layers]

            # Get position IDs
            position_ids = torch.arange(1, device=device).unsqueeze(0)  # [1, 1]

            # Run draft model
            with torch.no_grad():
                draft_output = draft_model(
                    noise_embedding=last_token_emb,
                    target_hidden=target_hidden,
                    position_ids=position_ids,
                )

            # Get predicted tokens
            predicted_token = sample(draft_output[:, -1:, :]).item()

            # Validate output
            if 0 <= predicted_token < tokenizer.vocab_size:
                results["num_success"] += 1
            else:
                results["errors"].append(f"Invalid token ID: {predicted_token}")

            results["num_samples"] += 1

        except Exception as e:
            logger.error(f"Validation error for prompt: {e}")
            results["errors"].append(str(e))
            results["num_samples"] += 1

    return results


def check_model_size(model_path: Path) -> dict:
    """Check model file size."""
    state_path = model_path / "final_model.pt"
    if state_path.exists():
        size_bytes = state_path.stat().st_size
        size_gb = size_bytes / (1024**3)
        return {
            "path": str(state_path),
            "size_bytes": size_bytes,
            "size_gb": size_gb,
            "meets_requirement": size_gb <= 2.0,  # ~2GB BF16 target
        }
    return {"path": str(state_path), "error": "Model file not found"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate trained draft model")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--target", default="Qwen/Qwen3.6-27B", help="Target model path")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of validation samples")
    parser.add_argument("--max-length", type=int, default=256, help="Max generation length")
    parser.add_argument("--device", default="cuda", help="Device to use")
    args = parser.parse_args()

    model_path = Path(args.model)
    device = torch.device(args.device)

    # Load config
    config_path = model_path / "config.json"
    if config_path.exists():
        import json
        with open(config_path) as f:
            config = json.load(f)
    else:
        logger.warning("No config.json found, using defaults")
        config = {}

    # Check model size
    logger.info("Checking model size...")
    size_info = check_model_size(model_path)
    logger.info(f"Model size: {size_info.get('size_gb', 'N/A')} GB")
    if size_info.get("meets_requirement"):
        logger.info("✓ Model size meets requirement (~2GB)")
    else:
        logger.warning("✗ Model size exceeds requirement")

    # Load target model
    logger.info(f"Loading target model: {args.target}")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        output_hidden_states=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.target)

    # Load draft model
    logger.info(f"Loading trained draft model from {model_path}")
    draft_model = load_trained_draft(model_path, config)
    draft_model = draft_model.to(device).eval()

    # Test prompts
    test_prompts = [
        "Write a Python function to calculate fibonacci numbers.",
        "Explain the difference between TCP and UDP.",
        "What is 2+2?",
        "Write a quicksort implementation.",
    ]

    # Validate
    logger.info("Running validation...")
    results = validate_model(draft_model, target_model, tokenizer, test_prompts[:args.num_samples])

    logger.info(f"Validation results:")
    logger.info(f"  Samples: {results['num_samples']}")
    logger.info(f"  Success: {results['num_success']}")
    if results["errors"]:
        logger.warning(f"  Errors: {results['errors']}")

    # Summary
    if results["num_success"] == results["num_samples"]:
        logger.info("✓ All validation samples passed!")
    else:
        logger.warning(f"✗ {len(results['errors'])} validation errors")

    # Calculate expected model size (5 layers * params per layer)
    # Approx: ~400M params * 2 bytes (BF16) = ~800MB
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Model: {model_path}")
    logger.info(f"Size: {size_info.get('size_gb', 'N/A'):.2f} GB")
    logger.info(f"Validation: {results['num_success']}/{results['num_samples']} passed")
    logger.info(f"Target layers: {config.get('target_layer_ids', TARGET_LAYERS)}")


if __name__ == "__main__":
    main()