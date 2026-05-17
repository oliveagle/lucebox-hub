#!/usr/bin/env python3
"""
训练 Qwen3.6 DFlash Draft 模型。

使用方法:
    python scripts/train_draft_qwen36.py \
        --data data/draft_training_qwen36.pt \
        --output models/draft/dflash-draft-3.6-trained \
        --epochs 10 \
        --batch-size 4 \
        --lr 1e-4
"""
import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3MLP,
)
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import rotate_half, eager_attention_forward, ALL_ATTENTION_FUNCTIONS

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))

# Target layer IDs for 27B model (64 layers)
TARGET_LAYERS = [1, 16, 31, 46, 60]


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Build target layer IDs for capturing hidden states."""
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class DFlashConfig:
    """DFlash draft model configuration."""
    hidden_size: int = 5120
    num_hidden_layers: int = 5
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 17408
    vocab_size: int = 152064
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000
    max_position_embeddings: int = 32768
    block_size: int = 16
    target_layer_ids: tuple[int, ...] = (1, 16, 31, 46, 60)
    num_target_layers: int = 64
    mask_token_id: int = 151644
    attention_dropout: float = 0.0
    layer_types: tuple[str, ...] = ("full_attention",) * 5
    sliding_window: Optional[int] = None
    use_cache: bool = True

    def to_qwen3_config(self) -> Qwen3Config:
        """Convert to Qwen3Config for compatibility."""
        return Qwen3Config(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            rms_norm_eps=self.rms_norm_eps,
            rope_theta=self.rope_theta,
            max_position_embeddings=self.max_position_embeddings,
            attention_dropout=self.attention_dropout,
            use_cache=self.use_cache,
        )


class Qwen3DFlashAttention(nn.Module):
    """DFlash attention with target hidden conditioning."""

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # Non-causal for denoising

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]

        # Project queries from noise (hidden_states)
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # Project keys/values from both target context and noise
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        # Concatenate context and noise for K, V
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=0.0 if not self.training else self.attention_dropout,
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3DFlashDecoderLayer(nn.Module):
    """DFlash decoder layer."""

    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config.to_qwen3_config())
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with target conditioning
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DFlashDraftModel(nn.Module):
    """DFlash Draft Model for training.

    This is a standalone training version that doesn't depend on Qwen3PreTrainedModel
    to avoid inheritance complexity during training.
    """

    def __init__(self, config: DFlashConfig) -> None:
        super().__init__()
        self.config = config

        # Decoder layers
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx)
             for layer_idx in range(config.num_hidden_layers)]
        )

        # Norm
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = Qwen3RotaryEmbedding(config.to_qwen3_config())

        # Target hidden projection
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection to vocab
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Block size (tokens per draft step)
        self.block_size = config.block_size

    def forward(
        self,
        noise_embedding: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            noise_embedding: [batch, block_size, hidden] - [last_token, MASK×(block_size-1)]
            target_hidden: [batch, 1, num_layers * hidden] - Captured target hidden states
            position_ids: [batch, block_size] - Position IDs

        Returns:
            logits: [batch, block_size, vocab_size]
        """
        hidden_states = noise_embedding

        # Project target hidden states
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        # Get rotary embeddings
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Pass through decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=position_embeddings,
            )

        # Final norm and projection
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


class DraftDataset(Dataset):
    """Draft training dataset."""

    def __init__(self, data_path: str, block_size: int = 16):
        self.data = torch.load(data_path)
        self.block_size = block_size
        self.target_layer_ids = TARGET_LAYERS

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.data[idx]

        # Input: last token from prompt
        input_ids = sample["input_ids"]
        last_token_id = input_ids[:, -1]  # [batch]

        # Target: next block_size tokens (excluding first which is last_token)
        target_ids = sample["target_ids"]
        target_block = target_ids[:, :self.block_size]  # [batch, block_size]

        # Hidden states: concat 5 layers at last position
        prompt_hidden = sample["prompt_hidden"]
        hidden_list = [
            prompt_hidden[f"layer_{i}"].squeeze(0)  # [1, 1, hidden] -> [1, hidden]
            for i in self.target_layer_ids
        ]
        target_hidden = torch.cat(hidden_list, dim=-1)  # [1, num_layers * hidden]

        return {
            "last_token_id": last_token_id.squeeze(0),  # [seq]
            "target_block": target_block.squeeze(0),  # [seq, block_size]
            "target_hidden": target_hidden.squeeze(0),  # [1, num_layers * hidden]
        }


class DraftTrainer:
    """Trainer for DFlash draft model."""

    def __init__(
        self,
        model: DFlashDraftModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        block_size: int = 16,
        gradient_accumulation_steps: int = 1,
        mixed_precision: bool = True,
    ):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.block_size = block_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.mixed_precision = mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            last_token_id = batch["last_token_id"][0].to(self.device)  # [1]
            target_block = batch["target_block"][0].to(self.device)  # [block_size]
            target_hidden = batch["target_hidden"][0].to(self.device).unsqueeze(0)  # [1, 1, num_layers * hidden]

            # Create noise embedding: [last_token, MASK×(block_size-1)]
            # For training, we use target tokens as "noised" input
            # First position is last_token (known), rest are to be denoised
            noise_ids = torch.cat([
                last_token_id.unsqueeze(0),
                target_block[:self.block_size - 1]
            ])  # [block_size]

            # Get embeddings (we'll create a simple embedding layer for training)
            # In practice, use target model's embedding layer
            noise_embedding = self.model.lm_head.weight[
                noise_ids
            ].unsqueeze(0)  # [1, block_size, hidden]

            # Create position IDs
            position_ids = torch.arange(
                self.block_size, device=self.device
            ).unsqueeze(0)  # [1, block_size]

            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    logits = self.model(
                        noise_embedding=noise_embedding,
                        target_hidden=target_hidden,
                        position_ids=position_ids,
                    )  # [1, block_size, vocab]

                    # Compute loss (skip position 0, predict positions 1 to block_size-1)
                    logits = logits[0, 1:self.block_size, :]  # [block_size-1, vocab]
                    loss_targets = target_block[1:self.block_size]  # [block_size-1]
                    loss = F.cross_entropy(logits, loss_targets, ignore_index=-100)
            else:
                logits = self.model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    position_ids=position_ids,
                )
                logits = logits[0, 1:self.block_size, :]
                loss_targets = target_block[1:self.block_size]
                loss = F.cross_entropy(logits, loss_targets, ignore_index=-100)

            # Gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            if self.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update weights
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            progress_bar.set_postfix({"loss": loss.item() * self.gradient_accumulation_steps})

        return total_loss / num_batches

    def save_checkpoint(self, output_dir: str, epoch: int, loss: float) -> None:
        """Save training checkpoint."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": {
                "hidden_size": self.config.hidden_size,
                "num_hidden_layers": self.config.num_hidden_layers,
                "num_attention_heads": self.config.num_attention_heads,
                "num_key_value_heads": self.config.num_key_value_heads,
                "intermediate_size": self.config.intermediate_size,
                "vocab_size": self.config.vocab_size,
                "block_size": self.config.block_size,
                "target_layer_ids": self.config.target_layer_ids,
            }
        }, output_path / f"checkpoint_epoch_{epoch}.pt")

        logger.info(f"Saved checkpoint to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DFlash draft model")
    parser.add_argument("--data", required=True, help="Path to collected data (.pt file)")
    parser.add_argument("--output", required=True, help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--block-size", type=int, default=16, help="Block size (tokens per draft step)")
    parser.add_argument("--gradient-accumulation", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--mixed-precision", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint-every", type=int, default=5, help="Save checkpoint every N epochs")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Create config
    config = DFlashConfig(
        hidden_size=5120,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=17408,
        vocab_size=152064,
        block_size=args.block_size,
        target_layer_ids=tuple(TARGET_LAYERS),
        num_target_layers=64,
    )

    logger.info(f"Creating DFlash draft model...")
    logger.info(f"Config: hidden_size={config.hidden_size}, num_layers={config.num_hidden_layers}")
    logger.info(f"  block_size={config.block_size}, target_layers={config.target_layer_ids}")

    model = DFlashDraftModel(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.1f} MB FP32)")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Load data
    logger.info(f"Loading data from {args.data}")
    dataset = DraftDataset(args.data, args.block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda x: x,  # Keep as list of dicts
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Trainer
    trainer = DraftTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        block_size=args.block_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
    )
    trainer.config = config

    # Training loop
    logger.info(f"Starting training for {args.epochs} epochs...")
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = trainer.train_epoch(epoch)
        logger.info(f"Epoch {epoch}/{args.epochs}: loss={epoch_loss:.4f}")

        # Save checkpoint
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            trainer.save_checkpoint(args.output, epoch, epoch_loss)

        # Track best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), Path(args.output) / "best_model.pt")
            logger.info(f"New best loss: {best_loss:.4f}")

        # Early stopping check
        if epoch_loss < 0.1:
            logger.info(f"Target loss < 0.1 reached at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    logger.info(f"Training completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    logger.info(f"Best loss: {best_loss:.4f}")

    # Save final model
    logger.info(f"Saving final model to {args.output}")
    torch.save(model.state_dict(), Path(args.output) / "final_model.pt")

    # Save config
    with open(Path(args.output) / "config.json", "w") as f:
        json.dump({
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "intermediate_size": config.intermediate_size,
            "vocab_size": config.vocab_size,
            "block_size": config.block_size,
            "target_layer_ids": list(config.target_layer_ids),
            "num_target_layers": config.num_target_layers,
        }, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()
