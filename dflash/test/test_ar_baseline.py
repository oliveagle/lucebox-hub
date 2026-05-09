#!/usr/bin/env python3
"""
测试 AR (autoregressive) 基线速度
"""
import subprocess
import struct
import os
import time
from transformers import AutoTokenizer

ROOT = "/home/oliveagle/repos/github.com/others/lucebox-hub/dflash"
TEST_DFLASH = f"{ROOT}/build/test_dflash"
TARGET_MODEL = f"{ROOT}/models/Qwen3.6-27B-Q4_K_M.gguf"

# HumanEval-style prompt
HUMANEVAL_PROMPT = '''from typing import List

def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """Check if any two numbers are closer to each other than threshold."""
    '''

def create_prompt(prompt_text, tokenizer, output_file):
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    with open(output_file, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return len(ids)

def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    prompt_file = "/tmp/humanevel_test_prompt.bin"
    output_file = "/tmp/humanevel_test_out.bin"

    n_prompt = create_prompt(HUMANEVAL_PROMPT, tokenizer, prompt_file)
    print(f"Prompt: {n_prompt} tokens")

    max_ctx = ((n_prompt + 256 + 64 + 255) // 256) * 256

    print("\n=== Testing AR (no draft) ===")
    start = time.time()
    result = subprocess.run(
        [
            TEST_DFLASH,
            TARGET_MODEL,
            "none",  # No draft model
            prompt_file,
            "256",
            output_file,
            f"--max-ctx={max_ctx}",
        ],
        capture_output=True, text=True, timeout=300
    )
    elapsed = time.time() - start

    print(f"Elapsed: {elapsed:.2f}s")
    print("\nOutput:")
    for line in result.stdout.splitlines():
        if "tok/s" in line or "prefill" in line:
            print(line)

    # Cleanup
    try:
        os.unlink(prompt_file)
        os.unlink(output_file)
    except:
        pass

if __name__ == "__main__":
    main()
