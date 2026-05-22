#!/usr/bin/env python3
"""
Convert TriAttention .pt stats file to .bin format for llama.cpp C library.

Usage:
    python convert_stats_to_bin.py --input stats.pt --output stats.bin

The .bin format:
- Header (64 bytes):
  - magic (4): 0x54524941 ("TRIA")
  - version (4): 2
  - num_layers (4)
  - num_heads (4): attention heads
  - num_kv_heads (4)
  - head_dim (4)
  - freq_count (4): head_dim / 2
  - rope_theta (4): float
  - attn_scale (4): float
  - reserved: rest of 64 bytes

- Layer budget scales (num_layers * 4): float per layer

- Per-head data (num_layers * num_heads * freq_count * 16):
  - q_mean_real (freq_count * 4)
  - q_mean_imag (freq_count * 4)
  - q_abs_mean (freq_count * 4)
  - mrl (freq_count * 4): reserved, write zeros
"""

import argparse
import struct
import sys
from pathlib import Path

import torch


TRIA_MAGIC = 0x54524941  # "TRIA"
TRIA_VERSION = 2
TRIA_HEADER_SIZE = 64


def convert_pt_to_bin(input_path: Path, output_path: Path) -> None:
    """Convert .pt stats to .bin format."""
    print(f"Loading .pt stats from: {input_path}")
    data = torch.load(input_path, map_location="cpu", weights_only=False)

    # Extract metadata
    metadata = data.get("metadata", {})
    stats_dict = data.get("stats", {})

    # Detect R-KV format (layerXX_headYY keys)
    is_rkv_format = any(
        k.startswith("layer") and "_head" in k
        for k in stats_dict.keys()
    )

    if is_rkv_format:
        # R-KV format: infer structure from keys
        layer_nums = set()
        head_nums = set()
        for key in stats_dict.keys():
            if key.startswith("layer") and "_head" in key:
                parts = key.split("_")
                if len(parts) == 2:
                    layer_nums.add(int(parts[0].replace("layer", "")))
                    head_nums.add(int(parts[1].replace("head", "")))

        num_layers = len(layer_nums)
        num_heads = len(head_nums)
        head_dim = metadata.get("head_dim", 256)
        rope_theta = metadata.get("rope_theta", 10000000.0)
    else:
        # TriAttention native format
        num_layers = metadata.get("num_layers", metadata.get("num_traces", 1))
        num_heads = metadata.get("num_attention_heads", 24)
        head_dim = metadata.get("head_dim", 256)
        rope_theta = metadata.get("rope_theta", 10000000.0)

    # For Qwen3.5/3.6-27B: num_kv_heads = 4 (GQA)
    num_kv_heads = metadata.get("num_kv_heads", 4)
    freq_count = head_dim // 2
    attn_scale = 1.0

    print(f"  num_layers: {num_layers}")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_dim: {head_dim}")
    print(f"  freq_count: {freq_count}")
    print(f"  rope_theta: {rope_theta}")

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        # Write header
        f.write(struct.pack("<I", TRIA_MAGIC))
        f.write(struct.pack("<I", TRIA_VERSION))
        f.write(struct.pack("<I", num_layers))
        f.write(struct.pack("<I", num_heads))
        f.write(struct.pack("<I", num_kv_heads))
        f.write(struct.pack("<I", head_dim))
        f.write(struct.pack("<I", freq_count))
        f.write(struct.pack("<f", rope_theta))
        f.write(struct.pack("<f", attn_scale))

        # Write reserved bytes to reach 64 bytes
        header_written = 4 * 8  # 8 fields of 4 bytes each
        f.write(b"\x00" * (TRIA_HEADER_SIZE - header_written))

        # Write layer budget scales (all 1.0 for now)
        for _ in range(num_layers):
            f.write(struct.pack("<f", 1.0))

        # Write per-head stats
        # Format: flat array [num_layers * num_heads]
        # Each entry: q_mean_real, q_mean_imag, q_abs_mean, mrl (zeros)
        for layer_idx in sorted(layer_nums if is_rkv_format else range(num_layers)):
            for head_idx in range(num_heads):
                if is_rkv_format:
                    key = f"layer{layer_idx:02d}_head{head_idx:02d}"
                    head_data = stats_dict.get(key, {})
                    q_mean_real = head_data.get("q_mean_real")
                    q_mean_imag = head_data.get("q_mean_imag")
                    q_abs_mean = head_data.get("q_abs_mean")
                else:
                    # TriAttention native format
                    layer_data = stats_dict.get(layer_idx, {})
                    q_mean_complex = layer_data.get("q_mean_complex")
                    q_abs_mean = layer_data.get("q_abs_mean")

                    if q_mean_complex is not None:
                        # q_mean_complex: [num_heads, freq_count, 2]
                        if head_idx < q_mean_complex.shape[0]:
                            q_mean_real = q_mean_complex[head_idx, :, 0]
                            q_mean_imag = q_mean_complex[head_idx, :, 1]
                        else:
                            q_mean_real = None
                            q_mean_imag = None
                    else:
                        q_mean_real = None
                        q_mean_imag = None

                    if q_abs_mean is not None and head_idx < q_abs_mean.shape[0]:
                        q_abs_mean = q_abs_mean[head_idx]
                    else:
                        q_abs_mean = None

                # Convert tensors to float arrays if needed
                def to_float_array(tensor):
                    if tensor is None:
                        return [0.0] * freq_count
                    if isinstance(tensor, torch.Tensor):
                        return tensor.cpu().numpy().tolist()[:freq_count]
                    return list(tensor)[:freq_count]

                q_mean_real = to_float_array(q_mean_real)
                q_mean_imag = to_float_array(q_mean_imag)
                q_abs_mean = to_float_array(q_abs_mean)

                # Pad to freq_count if needed
                q_mean_real.extend([0.0] * (freq_count - len(q_mean_real)))
                q_mean_imag.extend([0.0] * (freq_count - len(q_mean_imag)))
                q_abs_mean.extend([0.0] * (freq_count - len(q_abs_mean)))

                # Write arrays
                for val in q_mean_real:
                    f.write(struct.pack("<f", val))
                for val in q_mean_imag:
                    f.write(struct.pack("<f", val))
                for val in q_abs_mean:
                    f.write(struct.pack("<f", val))

                # Write mrl (reserved, zeros)
                for _ in range(freq_count):
                    f.write(struct.pack("<f", 0.0))

    print(f"Saved .bin stats to: {output_path}")
    print(f"File size: {output_path.stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TriAttention .pt stats to .bin format"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input .pt stats file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output .bin stats file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    try:
        convert_pt_to_bin(input_path, output_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
