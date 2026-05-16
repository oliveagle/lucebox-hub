#!/usr/bin/env python3
"""DFlash Chain vs DFlash+DDTree benchmark on gfx1151."""
import os, subprocess, struct, tempfile, time, re

TARGET = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3___5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
DRAFT  = "/mnt/eaget-4tb/modelscope_models/z-lab/Qwen3___6-27B-DFlash/model.safetensors"
DAEMON = "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/build/test_dflash"

env = {
    **os.environ,
    "LD_LIBRARY_PATH": "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:"
                       "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:"
                       + os.environ.get("LD_LIBRARY_PATH", ""),
}

def write_prompt(ids, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for t in ids:
            f.write(struct.pack("<i", t))

def run_dflash(prompt_path, n_gen, use_ddtree, budget=22):
    out_path = tempfile.mktemp(suffix=".bin")
    args = [DAEMON, TARGET, DRAFT, prompt_path, str(n_gen), out_path]
    if use_ddtree:
        args += ["--fast-rollback", "--ddtree", f"--ddtree-budget={budget}"]
    else:
        args += ["--fast-rollback"]

    t0 = time.time()
    proc = subprocess.run(
        args, text=True, timeout=600, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    elapsed = time.time() - t0

    # Parse output
    tok_s = 0.0
    accept_n = 0
    total = 0
    for line in proc.stdout.splitlines():
        m = re.search(r"(\d+\.\d+) tok/s", line)
        if m:
            tok_s = float(m.group(1))
        m = re.search(r"accepted=(\d+)/(\d+)", line)
        if m:
            accept_n = int(m.group(1))
            total = int(m.group(2))

    # Parse timing
    avg_sum = 0.0
    for line in proc.stdout.splitlines():
        m = re.search(r"sum\s+(\d+\.\d+)", line)
        if m:
            avg_sum = float(m.group(1))

    if os.path.exists(out_path):
        os.unlink(out_path)
    return {
        "tok_s": tok_s,
        "accept_n": accept_n,
        "total": total,
        "accept_pct": (accept_n / total * 100) if total > 0 else 0,
        "avg_step_ms": avg_sum,
        "wall_time": elapsed,
    }

print("=" * 90)
print("DFlash Chain vs DFlash+DDTree Benchmark — gfx1151")
print("=" * 90)

PROMPT_LEN = 200
N_GEN = 256
ROUNDS = 3

results_chain = []
results_ddtree = []

for r in range(ROUNDS):
    _, prompt_path = tempfile.mkstemp(suffix=".bin")
    # Write random-ish prompt
    ids = [(i * 137 + r * 251) % 248000 for i in range(PROMPT_LEN)]
    write_prompt(ids, prompt_path)

    print(f"\n--- Round {r+1}/{ROUNDS} ---")

    print("  DFlash (chain, no DDTree)...", end=" ", flush=True)
    r1 = run_dflash(prompt_path, N_GEN, use_ddtree=False)
    results_chain.append(r1)
    print(f"{r1['tok_s']:.1f} tok/s, accept={r1['accept_pct']:.1f}%")

    print("  DFlash (+ DDTree budget=22)...", end=" ", flush=True)
    r2 = run_dflash(prompt_path, N_GEN, use_ddtree=True, budget=22)
    results_ddtree.append(r2)
    print(f"{r2['tok_s']:.1f} tok/s, accept={r2['accept_pct']:.1f}%")

    os.unlink(prompt_path)

# Summary
avg_chain_toks = sum(r['tok_s'] for r in results_chain) / len(results_chain)
avg_chain_pct = sum(r['accept_pct'] for r in results_chain) / len(results_chain)
avg_chain_step = sum(r['avg_step_ms'] for r in results_chain) / len(results_chain)

avg_ddtree_toks = sum(r['tok_s'] for r in results_ddtree) / len(results_ddtree)
avg_ddtree_pct = sum(r['accept_pct'] for r in results_ddtree) / len(results_ddtree)
avg_ddtree_step = sum(r['avg_step_ms'] for r in results_ddtree) / len(results_ddtree)

print("\n" + "=" * 90)
print("Summary (3 rounds, 200 prompt + 256 generate tokens)")
print("=" * 90)
print(f"{'Mode':>30} | {'tok/s':>8} | {'Accept %':>8} | {'Avg step ms':>12}")
print("-" * 90)
print(f"{'DFlash (chain, no DDTree)':>30} | {avg_chain_toks:>8.1f} | {avg_chain_pct:>8.1f} | {avg_chain_step:>12.1f}")
print(f"{'DFlash (+ DDTree budget=22)':>30} | {avg_ddtree_toks:>8.1f} | {avg_ddtree_pct:>8.1f} | {avg_ddtree_step:>12.1f}")
print("=" * 90)

speedup = (avg_ddtree_toks / avg_chain_toks) if avg_chain_toks > 0 else 1.0
print(f"\nDDTree speedup: {speedup:.2f}x")
print(f"Note: These are synthetic prompts (random token IDs). Real text prompts")
print(f"      with natural distributions would show larger DDTree gains.")
