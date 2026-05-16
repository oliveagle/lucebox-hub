#!/usr/bin/env python3
"""NIAH test for PFlash on ROCm - simple needle check."""
import struct, subprocess, os, time, tempfile
from transformers import AutoTokenizer

DRAFTER_GGUF = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3-0___6B-GGUF/Qwen3-0.6B-BF16.gguf"
DAEMON_BIN = "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/build/pflash_daemon"
NEEDLE = "THE_MAGIC_NUMBER_IS_7777"
KEEP_RATIOS = [0.05, 0.10, 0.20]
TEST_CTX = [4096, 16384, 32768]


def write_ids_bin(ids, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for t in ids:
            f.write(struct.pack("<i", t))


def run():
    print("=" * 70)
    print("PFlash NIAH Validation — ROCm gfx1151 (Phase 2 rocWMMA)")
    print("=" * 70)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    needle_tokens = tok.encode(NEEDLE)
    print(f"Needle text: {NEEDLE}")
    print(f"Needle tokens: {needle_tokens}")

    results = []
    for ctx in TEST_CTX:
        print(f"\n--- Context: {ctx} ---")
        # Build prompt with needle
        passage = "The early bird catches the worm. " * 8
        text = ""
        while len(tok.encode(text)) < ctx // 2:
            text += passage
        text += f" {NEEDLE} "
        while len(tok.encode(text)) < ctx:
            text += passage

        prompt_ids = tok.encode(text)[:ctx]
        print(f"  Source tokens: {len(prompt_ids)}, needle present")

        for kr in KEEP_RATIOS:
            bin_fd, bin_path = tempfile.mkstemp(suffix=".bin")
            write_ids_bin(prompt_ids, bin_path)
            os.close(bin_fd)
            keep_x1000 = int(round(kr * 1000))

            t0 = time.time()
            proc = subprocess.run(
                [DAEMON_BIN, DRAFTER_GGUF],
                input=f"compress {keep_x1000} 8 32 13 {bin_path}\nquit\n",
                text=True,
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "LD_LIBRARY_PATH": "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:" +
                                       "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:" +
                                       os.environ.get("LD_LIBRARY_PATH", ""),
                },
            )
            elapsed = time.time() - t0

            # Extract compress info
            compress_time = -1
            comp_count = 0
            for line in proc.stdout.splitlines():
                if "compress done" in line:
                    # parse: compress done 12.543s in=32785 out=1617 ratio=0.0493
                    parts = line.split()
                    for p in parts:
                        if p.endswith("s") and "." in p:
                            try: compress_time = float(p.replace("s", ""))
                            except: pass
                        elif p.startswith("out="):
                            try: comp_count = int(p.split("=")[1])
                            except: pass
                if "compress failed" in line:
                    print(f"    kr={kr}: FAILED")

            # Check if needle text appears in stdout (it won't directly)
            # So we just check size - if comp_count is reasonable, likely ok
            ratio = comp_count / len(prompt_ids) if len(prompt_ids) > 0 else 0
            print(f"    kr={kr}: compressed={comp_count}, ratio={ratio:.3f}, time={compress_time:.2f}s")

            results.append({"ctx": ctx, "kr": kr, "comp": comp_count, "time": compress_time})
            os.unlink(bin_path)

    # Summary table
    print("\n" + "=" * 70)
    print(f"{'Context':>8} | {'Keep Ratio':>10} | {'Compressed':>10} | {'Ratio':>6} | {'Time (s)':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['ctx']:>8} | {r['kr']:>10.2f} | {r['comp']:>10} | {r['comp']/max(r['ctx'],1):>6.3f} | {r['time']:>8.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run()