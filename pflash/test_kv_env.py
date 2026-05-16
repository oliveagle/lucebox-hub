#!/usr/bin/env python3
"""Simple test to see KV type debug output."""
import os, subprocess, struct, tempfile

DRAFTER_GGUF = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3-0___6B-GGUF/Qwen3-0.6B-BF16.gguf"
DAEMON_BIN = "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/build/pflash_daemon"

# Small test prompt
prompt_ids = [1, 2, 3] * 4096  # 12K tokens

_, bin_path = tempfile.mkstemp(suffix=".bin")
with open(bin_path, "wb") as f:
    f.write(struct.pack("<I", len(prompt_ids)))
    for t in prompt_ids:
        f.write(struct.pack("<i", t))

env = {
    **os.environ,
    "LD_LIBRARY_PATH": "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:" +
                       "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:" +
                       os.environ.get("LD_LIBRARY_PATH", ""),
    "DFLASH27B_KV_K": "tq3_0",
    "DFLASH27B_KV_V": "q8_0",
}

cmd = f"echo 'compress 50 8 32 13 {bin_path}' | {DAEMON_BIN} {DRAFTER_GGUF}"
proc = subprocess.run(cmd, shell=True, text=True, timeout=300, env=env,
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE)

print("=== STDOUT ===")
for line in proc.stdout.splitlines():
    print(line)

print("\n=== STDERR ===")
for line in proc.stderr.splitlines():
    print(line)

os.unlink(bin_path)