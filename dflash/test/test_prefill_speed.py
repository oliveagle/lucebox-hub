#!/usr/bin/env python3
"""
测试 Qwen3.6-27B 在不同 context 下的 Prefill 速度
"""
import subprocess
import struct
import os
import time
import sys
from pathlib import Path

def read_int32_file(path):
    with open(path, "rb") as f:
        data = f.read()
    n = len(data) // 4
    return list(struct.unpack(f"<{n}i", data))

def write_int32_file(path, ids):
    with open(path, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))

def run_prefill_bench(prompt_bin, max_ctx, kv_type="q8_0", n_gen=32, label=""):
    """运行 prefill 基准测试"""
    dflash_root = Path("/home/oliveagle/repos/github.com/others/lucebox-hub/dflash")
    test_dflash = dflash_root / "build/test_dflash"
    target_gguf = dflash_root / "models/Qwen3.6-27B-Q4_K_M.gguf"
    draft_safetensors = dflash_root / "models/draft/model.safetensors"

    # 临时输出文件
    out_bin = f"/tmp/bench_{label}_out.bin"

    # 计算 max_ctx (对齐到 256)
    src_ids = read_int32_file(prompt_bin)
    prompt_len = len(src_ids)
    pad = 64
    auto_max_ctx = ((prompt_len + n_gen + pad + 255) // 256) * 256
    if max_ctx > 0:
        auto_max_ctx = max_ctx

    env = {**os.environ}
    env["DFLASH27B_KV_TQ3"] = "1" if kv_type == "tq3_0" else "0"

    print("=" * 60)
    print(f"Prefill 测试: {label}")
    print(f"  Prompt: {prompt_len} tokens")
    print(f"  Max ctx: {auto_max_ctx}")
    print(f"  KV type: {kv_type}")
    print("=" * 60)

    cmd = [
        str(test_dflash),
        str(target_gguf),
        str(draft_safetensors),
        prompt_bin,
        str(n_gen),
        out_bin,
        f"--max-ctx={auto_max_ctx}",
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
        print(f"超时！测试被中断")
        return None

    # 解析输出
    stdout = result.stdout
    stderr = result.stderr

    # 查找 prefill 时间
    import re

    # 格式: [prefill] N tokens in X.XX s, last_tok=XXXX
    m = re.search(r"\[prefill\] (\d+) tokens in ([\d.]+) s", stdout)
    if m:
        prefill_tokens = int(m.group(1))
        prefill_time = float(m.group(2))
        prefill_speed = prefill_tokens / prefill_time
    else:
        # 尝试其他格式
        m = re.search(r"prefill tokens=(\d+) time=([\d.]+) speed=([\d.]+)", stdout)
        if m:
            prefill_tokens = int(m.group(1))
            prefill_time = float(m.group(2))
            prefill_speed = float(m.group(3))
        else:
            # 使用总时间
            prefill_time = elapsed
            prefill_tokens = prompt_len
            prefill_speed = prefill_tokens / prefill_time

    print(f"Prefill tokens: {prefill_tokens}")
    print(f"Prefill time: {prefill_time:.2f}s")
    print(f"Prefill speed: {prefill_speed:.2f} tokens/s")
    print()

    # 打印相关日志
    for line in stdout.split('\n'):
        if any(x in line.lower() for x in ['prefill', 'timing', 'time']):
            print(f"  {line}")

    return {
        "label": label,
        "prompt_tokens": prompt_len,
        "max_ctx": auto_max_ctx,
        "kv_type": kv_type,
        "prefill_tokens": prefill_tokens,
        "prefill_time": prefill_time,
        "prefill_speed": prefill_speed,
    }

def main():
    dflash_root = Path("/home/oliveagle/repos/github.com/others/lucebox-hub/dflash")
    test_dflash = dflash_root / "build/test_dflash"

    if not test_dflash.exists():
        print(f"Error: test_dflash not found at {test_dflash}")
        sys.exit(1)

    results = []

    # 测试 64K
    result = run_prefill_bench(
        prompt_bin="/tmp/prompt_64k.bin",
        max_ctx=65536,
        kv_type="q8_0",
        label="64k_q8"
    )
    if result:
        results.append(result)

    # 测试 64K with TQ3
    result = run_prefill_bench(
        prompt_bin="/tmp/prompt_64k.bin",
        max_ctx=65536,
        kv_type="tq3_0",
        label="64k_tq3"
    )
    if result:
        results.append(result)

    # 测试 128K
    result = run_prefill_bench(
        prompt_bin="/tmp/prompt_128k.bin",
        max_ctx=131072,
        kv_type="tq3_0",
        label="128k_tq3"
    )
    if result:
        results.append(result)

    # 打印汇总
    print("\n" + "=" * 60)
    print("Prefill 速度测试结果汇总")
    print("=" * 60)
    print(f"{'测试':<15} {'Tokens':>8} {'时间(s)':>10} {'速度(tok/s)':>15}")
    print("-" * 60)
    for r in results:
        print(f"{r['label']:<15} {r['prompt_tokens']:>8} {r['prefill_time']:>10.2f} {r['prefill_speed']:>15.2f}")

    print()
    print(f"64K context 平均速度: {sum(r['prefill_speed'] for r in results if '64k' in r['label']) / len([r for r in results if '64k' in r['label']]):.2f} tokens/s")
    print(f"128K context 速度: {next((r['prefill_speed'] for r in results if '128k' in r['label']), 0):.2f} tokens/s")

if __name__ == "__main__":
    main()