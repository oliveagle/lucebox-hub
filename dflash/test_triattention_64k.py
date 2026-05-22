#!/usr/bin/env python3
"""Test TriAttention compression at 64K context.

This focuses on validating that:
1. TriAttention compression triggers correctly at 64K context
2. cache.cur_pos reduces after compression
3. Memory savings are measurable

Usage: python3 test_triattention_64k.py
"""

import subprocess
import os
import time
import sys

ROCM_PATH = os.path.expanduser("~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel")
LD_LIBRARY_PATH = os.path.expanduser("~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib") + ":" + \
                  os.path.expanduser("~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib")

TARGET_MODEL = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3___6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL = "/mnt/eaget-4tb/modelscope_models/z-lab/Qwen3___6-27B-DFlash/model.safetensors"
BINARY = os.path.join(os.path.dirname(__file__), "build-triatt", "test_dflash")
PROMPT_64K = "/tmp/long_prompt_65536.bin"

def run_test(tri_enabled, output_file):
    """Run a test with TriAttention on or off."""
    env = os.environ.copy()
    env.update({
        "ROCM_PATH": ROCM_PATH,
        "LD_LIBRARY_PATH": LD_LIBRARY_PATH,
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
        "TRIATTN_ENABLED": "1" if tri_enabled else "0",
    })

    if tri_enabled:
        env.update({
            "TRIATTN_KV_BUDGET": "4096",
            "TRIATTN_MIN_KEEP_RATIO": "0.75",
        })

    cmd = [
        BINARY,
        TARGET_MODEL,
        DRAFT_MODEL,
        PROMPT_64K,
        "256",
        output_file,
    ]

    print(f"Running: {'TRIATTENTION ON' if tri_enabled else 'DFlash only'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"TRIATTN_ENABLED={env.get('TRIATTN_ENABLED', '0')}")

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        elapsed = time.time() - start_time
        print(f"Exit code: {result.returncode}")
        print(f"Time: {elapsed:.1f}s")
        print(f"Stdout: {result.stdout}")
        print(f"Stderr: {result.stderr}")
        return result
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"Timeout after {elapsed:.1f}s")
        return None

def main():
    print("=" * 60)
    print("64K Context TriAttention Validation")
    print("=" * 60)

    # Check binary exists
    if not os.path.exists(BINARY):
        print(f"Binary not found: {BINARY}")
        print("Please build first: cmake --build build-triatt --target test_dflash")
        sys.exit(1)

    # Check prompt exists
    if not os.path.exists(PROMPT_64K):
        print(f"Prompt not found: {PROMPT_64K}")
        print("Please generate first: python3 scripts/generate_long_prompt.py /tmp/long_prompt_65536.bin 65536")
        sys.exit(1)

    print(f"\nBinary: {BINARY}")
    print(f"Target: {TARGET_MODEL}")
    print(f"Draft: {DRAFT_MODEL}")
    print(f"Prompt: {PROMPT_64K} ({os.path.getsize(PROMPT_64K)} bytes = {os.path.getsize(PROMPT_64K)//4} tokens)")
    print()

    # Test 1: DFlash only (baseline)
    print("\n--- Test 1: DFlash only (64K context) ---")
    run_test(False, "/tmp/dflash_only_64k_out.bin")

    # Test 2: DFlash + TriAttention
    print("\n--- Test 2: DFlash + TriAttention (64K context) ---")
    run_test(True, "/tmp/triattention_64k_out.bin")

    print("\n" + "=" * 60)
    print("Tests complete")
    print("=" * 60)

if __name__ == "__main__":
    main()
