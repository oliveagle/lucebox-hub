#!/usr/bin/env python3
"""
测试 64K 和 128K 的 Prefill 速度
"""
import subprocess
import struct
import os
import time
import sys
from pathlib import Path
import re
from collections import defaultdict

def read_int32_file(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}i", data))

def test_prefill(prompt_path, max_ctx, label):
    """直接测试单个 prompt 的 prefill"""
    dflash_root = Path("/home/oliveagle/repos/github.com/others/lucebox-hub/dflash")
    test_dflash = dflash_root / "build/test_dflash"
    target_gguf = dflash_root / "models/Qwen3.6-27B-Q4_K_M.gguf"
    draft_safetensors = dflash_root / "models/draft/model.safetensors"
    out_bin = f"/tmp/out_{label}.bin"

    prompt_ids = read_int32_file(prompt_path)
    prompt_len = len(prompt_ids)

    print(f"\n{'=' * 60}")
    print(f"测试 {label}: {prompt_len} tokens")
    print(f"{'=' * 60}")

    # 环境变量
    env = {**os.environ}
    # 使用 TQ3 KV 缓存以节省内存
    env["DFLASH27B_KV_TQ3"] = "1"

    cmd = [
        str(test_dflash),
        str(target_gguf),
        str(draft_safetensors),
        prompt_path,
        str(16),  # n_gen 只需要少量生成
        out_bin,
        f"--max-ctx={max_ctx}",
        "--fast-rollback",
    ]

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=600
        )
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        print(f"测试超时！")
        return None

    # 解析输出
    stdout = result.stdout
    stderr = result.stderr

    print("\nSTDOUT:")
    print("-" * 60)
    print(stdout)
    print("\nSTDERR:")
    print("-" * 60)
    print(stderr)

    # 查找 prefill 时间
    m = re.search(r"\[prefill\] (\d+) tokens in ([\d.]+) s", stdout)
    if m:
        prefill_tokens = int(m.group(1))
        prefill_time = float(m.group(2))
        prefill_speed = prefill_tokens / prefill_time
        print(f"\n{'=' * 60}")
        print(f"结果统计:")
        print(f"  Prefill tokens: {prefill_tokens}")
        print(f"  Prefill time: {prefill_time:.2f}s")
        print(f"  Prefill speed: {prefill_speed:.2f} tokens/s")
        print(f"{'=' * 60}")
        return {
            "label": label,
            "prompt_tokens": prompt_len,
            "max_ctx": max_ctx,
            "prefill_time": prefill_time,
            "prefill_speed": prefill_speed,
        }
    else:
        print(f"\n未能找到 prefill 时间，使用总时间 {elapsed:.2f}s")
        return {
            "label": label,
            "prompt_tokens": prompt_len,
            "max_ctx": max_ctx,
            "prefill_time": elapsed,
            "prefill_speed": prompt_len / elapsed,
        }

def clean_gpu():
    """清理 GPU 内存"""
    subprocess.run(["pkill", "-9", "test_dflash"], capture_output=True)
    subprocess.run(["pkill", "-9", "test_generate"], capture_output=True)
    subprocess.run(["pkill", "-9", "python"], capture_output=True)
    time.sleep(3)

def main():
    results = []

    # 测试 64K
    clean_gpu()
    result = test_prefill("/tmp/prompt_64k.bin", 65536, "64K-ctx")
    if result:
        results.append(result)

    # 测试 128K
    clean_gpu()
    result = test_prefill("/tmp/prompt_128k.bin", 131072, "128K-ctx")
    if result:
        results.append(result)

    # 汇总
    print(f"\n{'=' * 60}")
    print(f"Prefill 速度测试汇总")
    print(f"{'=' * 60}")
    print(f"{'Context':<15} {'Tokens':>10} {'Time(s)':>12} {'Speed(tok/s)':>15}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<15} {r['prompt_tokens']:>10} {r['prefill_time']:>12.2f} {r['prefill_speed']:>15.2f}")
    print(f"{'=' * 60}")

if __name__ == "__main__":
    main()