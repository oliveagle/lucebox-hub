#!/usr/bin/env python3
"""
验证 DFlash Draft GGUF 文件的完整性和正确性。

用法:
    python scripts/verify_draft_gguf.py models/draft/dflash-draft-3.6-q8_0.gguf
"""
import sys
from pathlib import Path

# Add gguf-py to path
repo_root = Path(__file__).parent.parent
gguf_path = repo_root / "deps" / "llama.cpp" / "gguf-py"
sys.path.insert(0, str(gguf_path))

import gguf
from pathlib import Path


def verify_gguf(gguf_path: Path) -> dict:
    """验证 GGUF 文件。"""
    print(f"[info] 验证 {gguf_path}")
    print(f"[info] 文件大小: {gguf_path.stat().st_size / 1024 / 1024:.1f} MB")

    reader = gguf.GGUFReader(gguf_path)

    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "info": {},
    }

    # 1. 验证架构
    arch_field = reader.fields.get("general.architecture")
    if not arch_field:
        results["errors"].append("缺少 general.architecture 元数据")
        results["valid"] = False
    else:
        arch = bytes(arch_field.parts[-1]).decode("utf-8", errors="ignore")
        results["info"]["architecture"] = arch
        if arch not in ("qwen35-dflash-draft", "dflash-draft"):
            results["warnings"].append(f"未知架构: {arch}")
        else:
            print(f"[ok] 架构: {arch}")

    # 2. 验证 DFlash 特定元数据
    required_metadata = {
        "qwen35-dflash-draft.block_count": 5,
        "qwen35-dflash-draft.embedding_length": 5120,
        "qwen35-dflash-draft.dflash.block_size": 16,
        "qwen35-dflash-draft.dflash.n_target_layers": 5,
    }

    for key, expected_value in required_metadata.items():
        field = reader.fields.get(key)
        if not field:
            results["errors"].append(f"缺少元数据: {key}")
            results["valid"] = False
        else:
            value = field.parts[-1]
            if value != expected_value:
                results["warnings"].append(
                    f"{key} = {value}, 预期 {expected_value}"
                )
            else:
                print(f"[ok] {key}: {value}")

    # 3. 验证关键张量
    required_tensors = [
        "dflash.fc.weight",
        "dflash.hidden_norm.weight",
        "output_norm.weight",
    ]

    tensor_names = [t.name for t in reader.tensors]
    for tensor_name in required_tensors:
        if tensor_name not in tensor_names:
            results["errors"].append(f"缺少张量: {tensor_name}")
            results["valid"] = False
        else:
            print(f"[ok] {tensor_name}: 存在")

    # 4. 验证层数
    layer_ids = set()
    for t in reader.tensors:
        if t.name.startswith("blk."):
            parts = t.name.split(".")
            if len(parts) >= 2:
                layer_ids.add(parts[1])

    n_layers = len(layer_ids)
    results["info"]["n_layers"] = n_layers
    if n_layers != 5:
        results["errors"].append(f"层数错误: {n_layers}, 预期 5")
        results["valid"] = False
    else:
        print(f"[ok] 层数: {n_layers}")

    # 5. 验证张量维度
    expected_shapes = {
        "dflash.fc.weight": [25600, 5120],
        "dflash.hidden_norm.weight": [5120],
        "output_norm.weight": [5120],
    }

    for tensor_name, expected_shape in expected_shapes.items():
        for t in reader.tensors:
            if t.name == tensor_name:
                actual_shape = list(t.shape)
                if actual_shape != expected_shape:
                    results["warnings"].append(
                        f"{tensor_name} 形状 {actual_shape} != 预期 {expected_shape}"
                    )
                else:
                    print(f"[ok] {tensor_name} 形状: {actual_shape}")
                break

    # 6. 验证量化类型
    q8_count = 0
    f32_count = 0
    for t in reader.tensors:
        tensor_type = t.tensor_type
        if tensor_type == 8:  # Q8_0
            q8_count += 1
        elif tensor_type == 0:  # F32
            f32_count += 1

    results["info"]["q8_tensors"] = q8_count
    results["info"]["f32_tensors"] = f32_count
    print(f"[ok] Q8_0 张量: {q8_count}")
    print(f"[ok] F32 张量: {f32_count}")

    # 7. 检查是否有过大的张量（可能未量化）
    for t in reader.tensors:
        if t.tensor_type not in (0, 8):  # 非 F32 或 Q8_0
            results["warnings"].append(
                f"{t.name} 使用了非常规量化类型: {t.tensor_type}"
            )

    return results


def main():
    if len(sys.argv) < 2:
        print("用法: python verify_draft_gguf.py <gguf_path>")
        sys.exit(1)

    gguf_path = Path(sys.argv[1])
    if not gguf_path.exists():
        print(f"[error] 文件不存在: {gguf_path}")
        sys.exit(1)

    results = verify_gguf(gguf_path)

    print("\n" + "=" * 50)
    if results["valid"]:
        print("[ok] GGUF 验证通过")
    else:
        print("[error] GGUF 验证失败")

    if results["warnings"]:
        print(f"\n[warn] 警告 ({len(results['warnings'])}):")
        for w in results["warnings"]:
            print(f"  - {w}")

    if results["errors"]:
        print(f"\n[error] 错误 ({len(results['errors'])}):")
        for e in results["errors"]:
            print(f"  - {e}")
        sys.exit(1)

    print("\n[info] 验证完成")
    sys.exit(0 if results["valid"] else 1)


if __name__ == "__main__":
    main()
