#!/usr/bin/env python3
"""
DFlash Draft 训练数据收集器。

支持多数据源并行收集，验证数据质量，保存为训练格式。

使用方法:
    # 收集 pilot 数据 (每个数据集 100 samples)
    python scripts/run_collect_draft_data.py --pilot

    # 收集完整数据 (默认配置)
    python scripts/run_collect_draft_data.py

    # 收集特定数据集
    python scripts/run_collect_draft_data.py --datasets humaneval math500

    # 指定模型路径 (本地 GGUF 模型需要转换)
    python scripts/run_collect_draft_data.py --model-path /path/to/Qwen3.6-27B
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add deps to path
sys.path.insert(0, str(Path(__file__).parent.parent / "deps" / "z-lab-dflash"))

TARGET_LAYERS = [1, 16, 31, 46, 60]


@dataclass
class CollectionConfig:
    """数据收集配置。"""
    # 输出目录
    output_dir: Path = field(default_factory=lambda: Path("models/training_data"))

    # 模型配置
    model_name: str = "Qwen/Qwen3.6-27B"
    model_path: Optional[str] = None

    # 数据集配置
    datasets: list[str] = field(default_factory=lambda: [
        "humaneval", "mbpp", "math500", "gsm8k", "sharegpt", "longpqa", "longalpaca"
    ])

    # 收集参数
    num_samples: int = 10000
    max_length: int = 2048
    max_new_tokens: int = 256

    # 质量控制
    validate: bool = True
    remove_nan: bool = True


def check_dependencies() -> dict[str, bool]:
    """检查依赖是否满足。"""
    deps = {
        "torch": False,
        "transformers": False,
        "datasets": False,
        "loguru": False,
    }

    try:
        import torch
        deps["torch"] = torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import transformers
        deps["transformers"] = True
    except ImportError:
        pass

    try:
        import datasets
        deps["datasets"] = True
    except ImportError:
        pass

    try:
        from loguru import logger
        deps["loguru"] = True
    except ImportError:
        pass

    return deps


def run_collection(
    config: CollectionConfig,
    pilot: bool = False,
) -> dict[str, any]:
    """运行数据收集。"""
    from loguru import logger

    # 确定模型路径
    model = config.model_path or config.model_name

    # 构建输出路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suffix = "_pilot" if pilot else ""
    output_path = config.output_dir / f"draft_training_qwen36{suffix}_{timestamp}.pt"

    # 构建命令
    cmd = [
        sys.executable,
        "scripts/collect_draft_data.py",
        "--model", model,
        "--output", str(output_path),
        "--num-samples", str(config.num_samples),
        "--max-length", str(config.max_length),
        "--max-new-tokens", str(config.max_new_tokens),
    ]

    # 添加数据集
    for ds in config.datasets:
        cmd.extend(["--dataset", ds])

    if pilot:
        cmd.append("--pilot-only")

    logger.info(f"Running collection command: {' '.join(cmd)}")

    # 运行收集
    start_time = time.time()
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, capture_output=False)
    elapsed = time.time() - start_time

    # 检查输出文件
    if output_path.exists():
        # 加载并验证数据
        import torch
        data = torch.load(output_path)
        meta_path = output_path.with_suffix(".meta.json")

        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {"num_samples": len(data)}

        return {
            "success": True,
            "output_path": str(output_path),
            "num_samples": len(data),
            "elapsed": elapsed,
            "metadata": metadata,
        }
    else:
        return {
            "success": False,
            "error": f"Output file not created: {output_path}",
            "elapsed": elapsed,
        }


def merge_data_files(
    input_files: list[Path],
    output_path: Path,
) -> dict[str, any]:
    """合并多个数据文件。"""
    import torch
    from loguru import logger

    all_data = []
    source_counts = {}

    for input_file in input_files:
        logger.info(f"Loading {input_file}...")
        data = torch.load(input_file)
        all_data.extend(data)

        # Count by source
        for sample in data:
            source = sample.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

    # Save merged data
    logger.info(f"Saving {len(all_data)} samples to {output_path}")
    torch.save(all_data, output_path)

    # Save metadata
    metadata = {
        "num_samples": len(all_data),
        "target_layers": TARGET_LAYERS,
        "source_counts": source_counts,
        "input_files": [str(f) for f in input_files],
    }

    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {
        "num_samples": len(all_data),
        "source_counts": source_counts,
    }


def main() -> None:
    from loguru import logger

    parser = argparse.ArgumentParser(description="DFlash Draft training data collector")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run pilot collection with reduced samples",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific datasets to collect",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--model-path",
        help="Path to local model (overrides model-name)",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3.6-27B",
        help="HuggingFace model name",
    )
    parser.add_argument(
        "--output-dir",
        default="models/training_data",
        help="Output directory for training data",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        help="Merge existing data files",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check dependencies and exit",
    )
    args = parser.parse_args()

    # Check dependencies
    logger.info("Checking dependencies...")
    deps = check_dependencies()

    missing = [k for k, v in deps.items() if not v]
    if missing:
        logger.error(f"Missing dependencies: {missing}")
        logger.info("Install with: pip install torch transformers datasets loguru")
        if not deps.get("torch"):
            logger.error("CUDA not available. Data collection requires GPU.")
        sys.exit(1)

    logger.info("All dependencies satisfied.")
    if args.check_only:
        sys.exit(0)

    # Merge mode
    if args.merge:
        logger.info("Merging data files...")
        input_files = [Path(f) for f in args.merge]
        output_path = Path(args.output_dir) / f"draft_training_qwen36_merged_{int(time.time())}.pt"
        result = merge_data_files(input_files, output_path)
        logger.info(f"Merged {result['num_samples']} samples")
        logger.info(f"Source counts: {result['source_counts']}")
        sys.exit(0)

    # Collection mode
    config = CollectionConfig(
        output_dir=Path(args.output_dir),
        model_name=args.model_name,
        model_path=args.model_path,
        datasets=args.datasets or config.datasets,
        num_samples=args.num_samples,
    )

    logger.info(f"Starting data collection...")
    logger.info(f"Model: {config.model_path or config.model_name}")
    logger.info(f"Datasets: {config.datasets}")
    logger.info(f"Output dir: {config.output_dir}")

    result = run_collection(config, pilot=args.pilot)

    if result["success"]:
        logger.info(f"Collection complete!")
        logger.info(f"  Output: {result['output_path']}")
        logger.info(f"  Samples: {result['num_samples']}")
        logger.info(f"  Time: {result['elapsed']:.1f}s")

        if "metadata" in result:
            meta = result["metadata"]
            if "validation" in meta:
                v = meta["validation"]
                logger.info(f"  Valid: {v['valid']}/{v['total']} ({v['valid_rate']:.1%})")

        logger.info("Next step: Run training with scripts/train_draft_qwen36.py")
    else:
        logger.error(f"Collection failed: {result['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
