#!/usr/bin/env python3
"""
导出训练好的 DFlash draft 模型到 HuggingFace 格式。

使用方法:
    python scripts/export_draft_hf.py \
        --checkpoint models/draft/dflash-draft-3.6-trained/checkpoint_epoch_10.pt \
        --output models/draft/dflash-draft-3.6-trained-hf
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from loguru import logger
from safetensors.torch import save_file
from transformers import Qwen3Config

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))
from dflash.model import DFlashDraftModel

# Target layer IDs
TARGET_LAYERS = [1, 16, 31, 46, 60]


def export_to_huggingface(
    checkpoint_path: str,
    output_dir: str,
    config_path: str = None,
) -> None:
    """Export trained checkpoint to HuggingFace format."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Get config
    config_data = checkpoint.get("config", {})
    if config_path:
        logger.info(f"Loading config from {config_path}")
        with open(config_path) as f:
            file_config = json.load(f)
        config_data.update(file_config)

    # Create Qwen3Config
    target_layer_ids = config_data.get("target_layer_ids", TARGET_LAYERS)
    qwen_config = Qwen3Config(
        hidden_size=config_data.get("hidden_size", 5120),
        num_hidden_layers=config_data.get("num_hidden_layers", 5),
        num_attention_heads=config_data.get("num_attention_heads", 32),
        num_key_value_heads=config_data.get("num_key_value_heads", 8),
        intermediate_size=config_data.get("intermediate_size", 17408),
        vocab_size=config_data.get("vocab_size", 152064),
        rms_norm_eps=1e-6,
        rope_theta=1_000_000,
        max_position_embeddings=32768,
        use_cache=True,
    )

    # Add DFlash-specific config
    qwen_config.dflash_config = {
        "target_layer_ids": target_layer_ids,
        "mask_token_id": 151644,
    }
    qwen_config.block_size = config_data.get("block_size", 16)
    qwen_config.num_target_layers = config_data.get("num_target_layers", 64)

    # Save config
    qwen_config.save_pretrained(output_dir)
    logger.info(f"Saved config to {output_dir}/config.json")

    # Save config.json with DFlash metadata
    hf_config_path = output_path / "config.json"
    with open(hf_config_path) as f:
        hf_config = json.load(f)

    hf_config.update({
        "dflash_config": {
            "target_layer_ids": target_layer_ids,
            "mask_token_id": 151644,
        },
        "block_size": config_data.get("block_size", 16),
        "num_target_layers": config_data.get("num_target_layers", 64),
        "model_type": "qwen3",
    })
    with open(hf_config_path, "w") as f:
        json.dump(hf_config, f, indent=2)

    # Extract model state dict
    model_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(model_state, dict):
        logger.error("Invalid model state format")
        return

    # Convert to safetensors
    logger.info("Converting to safetensors format...")
    save_file(model_state, output_path / "model.safetensors")

    # Create model card
    model_card = f"""# DFlash Draft Model

## Description
DFlash draft model trained for Qwen3.6-27B speculative decoding.

## Configuration
- Hidden size: {config_data.get('hidden_size', 5120)}
- Num layers: {config_data.get('num_hidden_layers', 5)}
- Num attention heads: {config_data.get('num_attention_heads', 32)}
- Num key-value heads: {config_data.get('num_key_value_heads', 8)}
- Intermediate size: {config_data.get('intermediate_size', 17408)}
- Vocab size: {config_data.get('vocab_size', 152064)}
- Block size: {config_data.get('block_size', 16)}
- Target layers: {target_layer_ids}

## Training
- Epoch: {checkpoint.get('epoch', 'N/A')}
- Final loss: {checkpoint.get('loss', 'N/A')}

## Usage
```python
from transformers import AutoModelForCausalLM
from dflash.model import DFlashDraftModel

# Load draft model
draft_model = DFlashDraftModel.from_pretrained("{output_dir}")
```
"""
    with open(output_path / "README.md", "w") as f:
        f.write(model_card)

    logger.info(f"Exported model to {output_dir}")
    logger.info(f"Files created:")
    logger.info(f"  - {output_dir}/config.json")
    logger.info(f"  - {output_dir}/model.safetensors")
    logger.info(f"  - {output_dir}/README.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DFlash draft to HuggingFace format")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--config", help="Path to config.json file")
    args = parser.parse_args()

    export_to_huggingface(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()