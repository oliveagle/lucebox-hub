#!/usr/bin/env python3
"""
验证训练后的 DFlash Draft 模型质量。

功能:
1. 加载训练后的 checkpoint
2. 构造测试样本 (模拟目标 hidden states + 噪声)
3. 验证 forward pass 输出格式正确
4. 验证 top-1 预测准确率 (随机样本)
5. 验证输出分布合理性

用法:
    # 验证训练后的 checkpoint
    python scripts/verify_trained_draft.py --checkpoint models/draft/dflash-draft-3.6-trained/best_model.pt

    # 验证指定配置下的推理
    python scripts/verify_trained_draft.py --checkpoint models/draft/dflash-draft-3.6-trained/best_model.pt --data data/draft_training_qwen36.pt --n-test 50
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))

from train_draft_qwen36 import DFlashConfig, DFlashDraftModel, TARGET_LAYERS


def load_trained_model(checkpoint_path: str, device: torch.device):
    """加载训练后的模型 checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = DFlashConfig(
        hidden_size=5120,
        num_hidden_layers=5,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=17408,
        vocab_size=152064,
        block_size=16,
        target_layer_ids=tuple(TARGET_LAYERS),
        num_target_layers=64,
    )

    model = DFlashDraftModel(config).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", None)
        print(f"[ok] 加载 checkpoint (epoch={epoch}, loss={loss:.4f})")
    else:
        model.load_state_dict(ckpt)
        print("[ok] 加载 model state dict")

    model.eval()
    return model, config


def test_output_format(model, device: torch.device, block_size: int = 16):
    """验证 forward pass 输出格式."""
    batch_size = 1
    hidden = model.fc.in_features  # num_target_layers * hidden_size

    # 构造测试输入
    noise_embedding = torch.randn(batch_size, block_size, 5120, device=device)
    target_hidden = torch.randn(batch_size, 1, hidden, device=device)
    position_ids = torch.arange(block_size, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
        )

    errors = []
    info = {}

    # 检查输出形状
    info["logits_shape"] = list(logits.shape)
    expected_shape = [batch_size, block_size, 152064]
    if list(logits.shape) != expected_shape:
        errors.append(f"logits 形状 {list(logits.shape)} != 预期 {expected_shape}")
    else:
        print(f"[ok] 输出形状: {list(logits.shape)}")

    # 检查是否有 NaN/Inf
    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    if has_nan:
        errors.append("logits 包含 NaN")
    else:
        print("[ok] 无 NaN")

    if has_inf:
        errors.append("logits 包含 Inf")
    else:
        print("[ok] 无 Inf")

    # 检查值范围
    info["logits_mean"] = logits.mean().item()
    info["logits_std"] = logits.std().item()
    info["logits_min"] = logits.min().item()
    info["logits_max"] = logits.max().item()

    print(f"[info] logits: mean={info['logits_mean']:.2f}, "
          f"std={info['logits_std']:.2f}, "
          f"min={info['logits_min']:.2f}, max={info['logits_max']:.2f}")

    return errors, info


def test_prediction_accuracy(model, config, data_path: str, device: torch.device,
                             n_test: int = 50, block_size: int = 16):
    """在测试样本上验证 top-1 预测准确率."""
    data = torch.load(data_path)

    if len(data) < n_test:
        print(f"[warn] 数据不足 ({len(data)} < {n_test})，使用全部样本")
        n_test = len(data)

    top1_correct = 0
    top5_correct = 0
    total_tokens = 0
    top1_total = 0

    with torch.no_grad():
        for i in range(n_test):
            sample = data[i]

            last_token_id = sample["input_ids"][:, -1]  # [batch]
            target_block = sample["target_ids"][:, :block_size]  # [batch, block_size]

            prompt_hidden = sample["prompt_hidden"]
            hidden_list = [
                prompt_hidden[f"layer_{lid}"].squeeze(0)
                for lid in TARGET_LAYERS
            ]
            target_hidden = torch.cat(hidden_list, dim=-1).unsqueeze(0)  # [1, 1, hidden*layers]

            # 构造输入: 使用真实 target tokens 作为噪声输入
            noise_ids = torch.cat([
                last_token_id,
                target_block[0, :block_size - 1]
            ])  # [block_size]
            noise_embedding = model.lm_head.weight[noise_ids].unsqueeze(0)
            position_ids = torch.arange(block_size, device=device).unsqueeze(0)

            logits = model(
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
                position_ids=position_ids,
            )  # [1, block_size, vocab]

            # 预测 (跳过 position 0)
            pred_logits = logits[0, 1:block_size, :]  # [block_size-1, vocab]
            targets = target_block[0, 1:block_size]  # [block_size-1]

            pred_ids = pred_logits.argmax(dim=-1)
            top5_ids = pred_logits.topk(5, dim=-1).indices

            top1_correct += (pred_ids == targets).sum().item()
            top1_total += len(targets)

            # Top-5 准确率
            for j, target in enumerate(targets):
                if target in top5_ids[j]:
                    top5_correct += 1

            total_tokens += len(targets)

    top1_rate = top1_correct / top1_total if top1_total > 0 else 0
    top5_rate = top5_correct / total_tokens if total_tokens > 0 else 0

    info = {
        "top1_accuracy": round(top1_rate, 4),
        "top5_accuracy": round(top5_rate, 4),
        "n_samples": n_test,
        "n_tokens_evaluated": total_tokens,
    }

    print(f"\n[info] 预测准确率 (n={n_test} samples, {total_tokens} tokens):")
    print(f"  Top-1: {top1_rate:.2%}")
    print(f"  Top-5: {top5_rate:.2%}")

    # 合理性检查
    errors = []
    if top1_rate < 0.01:
        errors.append(f"Top-1 准确率过低 ({top1_rate:.2%})，模型可能未训练或退化")

    return errors, info


def test_output_distribution(model, device: torch.device, block_size: int = 16):
    """验证输出概率分布是否合理."""
    batch_size = 1
    hidden = model.fc.in_features

    noise_embedding = torch.randn(batch_size, block_size, 5120, device=device)
    target_hidden = torch.randn(batch_size, 1, hidden, device=device)
    position_ids = torch.arange(block_size, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            position_ids=position_ids,
        )

    # 检查 softmax 分布
    probs = F.softmax(logits[0, 1, :], dim=-1)  # 取 position 1

    info = {
        "entropy": -(probs * (probs + 1e-10).log()).sum().item(),
        "max_prob": probs.max().item(),
        "top10_prob": probs.topk(10).values.sum().item(),
    }

    errors = []
    print(f"\n[info] 输出分布分析 (position 1):")
    print(f"  Entropy: {info['entropy']:.2f} bits")
    print(f"  Max probability: {info['max_prob']:.4f}")
    print(f"  Top-10 cumulative probability: {info['top10_prob']:.4f}")

    # 检查是否退化成均匀分布
    vocab_size = 152064
    uniform_entropy = torch.log(torch.tensor(vocab_size, dtype=torch.float32)).item()
    if info["entropy"] > uniform_entropy * 0.95:
        errors.append(f"输出分布接近均匀 (entropy={info['entropy']:.2f})，模型可能未训练")

    # 检查是否退化成单点分布
    if info["max_prob"] > 0.99:
        errors.append(f"输出退化为单点 (max_prob={info['max_prob']:.4f})，模型可能过拟合")

    return errors, info


def main():
    parser = argparse.ArgumentParser(description="验证训练后的 DFlash Draft 模型质量")
    parser.add_argument("--checkpoint", required=True, help="训练后的模型 checkpoint 路径")
    parser.add_argument("--data", help="训练数据路径 (用于准确率测试)")
    parser.add_argument("--n-test", type=int, default=50, help="测试样本数")
    parser.add_argument("--device", default="cuda", help="测试设备")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    try:
        model, config = load_trained_model(args.checkpoint, device)
    except Exception as e:
        print(f"[error] 加载模型失败: {e}")
        sys.exit(1)

    all_errors = []
    all_info = {}

    # 1. 输出格式验证
    print("\n" + "=" * 50)
    print("1. 输出格式验证")
    print("=" * 50)
    errors, info = test_output_format(model, device, config.block_size)
    all_errors.extend(errors)
    all_info.update(info)

    # 2. 输出分布验证
    print("\n" + "=" * 50)
    print("2. 输出分布验证")
    print("=" * 50)
    errors, info = test_output_distribution(model, device, config.block_size)
    all_errors.extend(errors)
    all_info.update(info)

    # 3. 预测准确率 (如果有数据)
    if args.data and Path(args.data).exists():
        print("\n" + "=" * 50)
        print("3. 预测准确率验证")
        print("=" * 50)
        errors, info = test_prediction_accuracy(
            model, config, args.data, device, args.n_test, config.block_size
        )
        all_errors.extend(errors)
        all_info.update(info)
    elif args.data:
        print(f"\n[warn] 数据文件不存在: {args.data}，跳过准确率测试")
    else:
        print("\n[skip] 未提供数据路径，跳过准确率测试")

    # 总结
    print("\n" + "=" * 50)
    print("验证结果总结")
    print("=" * 50)
    if all_errors:
        print(f"\n[error] {len(all_errors)} 个问题:")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n[ok] 所有验证通过")
        print(f"\n验证指标:")
        for k, v in all_info.items():
            print(f"  {k}: {v}")
        sys.exit(0)


if __name__ == "__main__":
    main()
