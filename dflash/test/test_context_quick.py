#!/usr/bin/env python3
"""
快速测试不同上下文大小的 prefill 和 decode 速度
"""
import subprocess
import struct
import os
import time
from pathlib import Path
from transformers import AutoTokenizer


ROOT = Path("/home/oliveagle/repos/github.com/others/lucebox-hub/dflash")
TEST_DFLASH = ROOT / "build/test_dflash"
TARGET_MODEL = ROOT / "models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL = ROOT / "models/draft/model.safetensors"


def create_prompt_file(target_tokens: int, output_file: str, tokenizer) -> int:
    """创建指定 token 数量的 prompt 文件"""
    base_text = "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。注意力机制允许模型动态关注输入的不同部分。"
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    repeat = (target_tokens // len(base_ids)) + 1
    test_ids = (base_ids * repeat)[:target_tokens]
    with open(output_file, "wb") as f:
        for t in test_ids:
            f.write(struct.pack("<i", int(t)))
    return len(test_ids)


def run_test(target_tokens: int, n_gen: int = 128):
    """运行测试并返回性能数据"""
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

    # 创建 prompt 文件
    prompt_file = f"/tmp/test_prompt_{target_tokens//1024}k.bin"
    out_file = f"/tmp/test_out_{target_tokens//1024}k.bin"

    actual_tokens = create_prompt_file(target_tokens, prompt_file, tokenizer)

    # 计算 max_ctx
    pad = 256
    run_max_ctx = ((actual_tokens + n_gen + pad + 255) // 256) * 256

    print(f"\n{'='*70}")
    print(f"测试 {target_tokens//1024}K context ({actual_tokens} tokens)")
    print(f"{'='*70}")

    cmd = [
        str(TEST_DFLASH),
        str(TARGET_MODEL),
        str(DRAFT_MODEL),
        prompt_file,
        str(n_gen),
        out_file,
        f"--max-ctx={run_max_ctx}",
        "--fast-rollback",
        "--ddtree",
        "--ddtree-budget=22"
    ]

    env = dict(os.environ)
    env["DFLASH27B_KV_TQ3"] = "1"

    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=600)
    total_time = time.time() - start

    stdout = result.stdout
    stderr = result.stderr

    # 解析输出
    import re
    prefill_match = re.search(r"\[prefill\]\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+s", stdout)
    decode_match = re.search(r"\[dflash\]\s+generated\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+s\s+->\s+([\d.]+)\s+tok/s", stdout)
    commit_match = re.search(r"avg commit/step=([\d.]+)", stdout)
    accept_match = re.search(r"accepted=(\d+)/(\d+)", stdout)

    prefill_tokens = int(prefill_match.group(1)) if prefill_match else 0
    prefill_time = float(prefill_match.group(2)) if prefill_match else 0
    decode_tokens = int(decode_match.group(1)) if decode_match else 0
    decode_time = float(decode_match.group(2)) if decode_match else 0
    decode_speed_from_log = float(decode_match.group(3)) if decode_match else None

    prefill_speed = prefill_tokens / prefill_time if prefill_time > 0 else 0
    decode_speed = decode_speed_from_log if decode_speed_from_log else (decode_tokens / decode_time if decode_time > 0 else 0)
    avg_commit = float(commit_match.group(1)) if commit_match else 0

    accept_rate = 0
    if accept_match:
        accepted = int(accept_match.group(1))
        total_draft = int(accept_match.group(2))
        accept_rate = (accepted / total_draft * 100) if total_draft > 0 else 0

    print(f"Prefill: {prefill_tokens} tokens in {prefill_time:.2f}s = {prefill_speed:.2f} tok/s")
    print(f"Decode: {decode_tokens} tokens in {decode_time:.2f}s = {decode_speed:.2f} tok/s")
    print(f"Avg commit: {avg_commit:.2f}")
    print(f"Accept rate: {accept_rate:.1f}%")

    # 清理
    try:
        os.unlink(prompt_file)
        os.unlink(out_file)
    except:
        pass

    return {
        "tokens": target_tokens,
        "actual_tokens": actual_tokens,
        "prefill_time": prefill_time,
        "prefill_speed": prefill_speed,
        "decode_time": decode_time,
        "decode_speed": decode_speed,
        "avg_commit": avg_commit,
        "accept_rate": accept_rate
    }


def main():
    print("="*70)
    print("DFlash 性能测试 - Prefill & Decode (直接使用 test_dflash)")
    print("="*70)

    # 清理
    subprocess.run(["pkill", "-9", "test_dflash"], capture_output=True)
    time.sleep(2)

    # 测试 4K - 256K
    test_sizes = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    results = []

    for size in test_sizes:
        try:
            result = run_test(size, n_gen=128)
            results.append(result)
            # 清理
            subprocess.run(["pkill", "-9", "test_dflash"], capture_output=True)
            time.sleep(2)
        except Exception as e:
            print(f"错误: {e}")
            subprocess.run(["pkill", "-9", "test_dflash"], capture_output=True)
            time.sleep(3)

    # 汇总
    print("\n\n" + "="*70)
    print("性能测试结果汇总")
    print("="*70)
    print(f"{'Context':>10} {'Prefill(s)':>12} {'Prefill(T/s)':>14} {'Decode(T/s)':>14} {'Commit':>8} {'Accept%':>8}")
    print("-"*70)
    for r in results:
        print(f"{r['tokens']//1024:>8}K {r['prefill_time']:>12.2f} {r['prefill_speed']:>14.2f} {r['decode_speed']:>14.2f} {r['avg_commit']:>8.2f} {r['accept_rate']:>8.1f}")
    print("="*70)


if __name__ == "__main__":
    main()
