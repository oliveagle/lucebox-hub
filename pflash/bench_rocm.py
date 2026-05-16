#!/usr/bin/env python3
"""PFlash performance benchmark with VRAM tracking for ROCm/gfx1151."""
import ctypes, struct, subprocess, os, time, tempfile
from transformers import AutoTokenizer
import threading

DRAFTER_GGUF = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3-0___6B-GGUF/Qwen3-0.6B-BF16.gguf"
DAEMON_BIN = "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/build/pflash_daemon"

TEST_CTX = [4096, 8192, 16384, 32768, 65536]
KEEP_RATIO = 0.05

# HIP runtime for VRAM monitoring
_hip = ctypes.CDLL('libamdhip64.so.7')


def get_vram_mb():
    """Get VRAM usage in MB via HIP API."""
    try:
        total = ctypes.c_size_t()
        free = ctypes.c_size_t()
        err = _hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if err == 0:
            used_mb = (total.value - free.value) // (1024 * 1024)
            return used_mb
    except Exception:
        pass
    return None


def write_ids_bin(ids, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(ids)))
        for t in ids:
            f.write(struct.pack("<i", t))


def run():
    print("=" * 90)
    print("PFlash Performance Benchmark — ROCm gfx1151 (with VRAM tracking)")
    print("=" * 90)

    # Check total VRAM
    total_vram = None
    try:
        total = ctypes.c_size_t()
        free = ctypes.c_size_t()
        err = _hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        if err == 0:
            total_vram = total.value // (1024 * 1024)
            print(f"Total VRAM: {total_vram} MB ({total_vram // 1024} GB)")
            print(f"Used VRAM:  {(total.value - free.value) // (1024**2)} MB (idle)")
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    results = []

    for ctx in TEST_CTX:
        print(f"\n{'='*60}")
        print(f"Context: {ctx} tokens")
        print(f"{'='*60}")

        # Generate prompt
        passage = (
            "The quick brown fox jumps over the lazy dog. "
            "In the beginning, the universe was created. "
            "This is a test of long context processing. "
            "Machine learning is transforming the world."
        )
        text = ""
        while len(tokenizer.encode(text)) < ctx:
            text += passage
        prompt_ids = tokenizer.encode(text)[:ctx]
        src_tokens = len(prompt_ids)

        # Write binary file
        bin_fd, bin_path = tempfile.mkstemp(suffix=".bin")
        write_ids_bin(prompt_ids, bin_path)
        os.close(bin_fd)

        keep_x1000 = int(round(KEEP_RATIO * 1000))

        # Baseline VRAM
        vram_before = get_vram_mb()
        print(f"VRAM (idle): {vram_before} MB" if vram_before else "VRAM: N/A")

        try:
            t0 = time.time()
            vram_samples = []

            # VRAM sampling thread
            stop_sampling = False

            def sample_vram():
                while not stop_sampling:
                    v = get_vram_mb()
                    if v is not None:
                        vram_samples.append(v)
                    time.sleep(0.2)

            sampler = threading.Thread(target=sample_vram, daemon=True)
            sampler.start()

            # Run daemon
            cmd = f"echo 'compress {keep_x1000} 8 32 13 {bin_path}' | {DAEMON_BIN} {DRAFTER_GGUF}"
            proc = subprocess.run(
                cmd,
                shell=True,
                text=True,
                timeout=600,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "LD_LIBRARY_PATH": "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:" +
                                       "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:" +
                                       os.environ.get("LD_LIBRARY_PATH", ""),
                },
            )

            stop_sampling = True
            sampler.join(timeout=2)
            elapsed = time.time() - t0

            # Parse output
            comp_count = 0
            for line in proc.stdout.splitlines():
                if "compress done" in line:
                    print(f"  {line.strip()}")
                    # parse out=XXXX
                    for p in line.split():
                        if p.startswith("out="):
                            try:
                                comp_count = int(p.split("=")[1])
                            except:
                                pass
                elif "ready" in line and "load=" in line:
                    print(f"  {line.strip()}")

            vram_peak = max(vram_samples) if vram_samples else None
            vram_after = get_vram_mb()

            if vram_peak:
                print(f"\n  VRAM peak: {vram_peak} MB ({vram_peak // 1024} GB)")
            if vram_before and vram_peak:
                print(f"  VRAM delta: {vram_peak - vram_before} MB")
            if vram_after:
                print(f"  VRAM after: {vram_after} MB")

            results.append({
                "ctx": ctx,
                "src": src_tokens,
                "comp": comp_count,
                "time": elapsed,
                "vram_before": vram_before,
                "vram_peak": vram_peak,
                "vram_delta": vram_peak - vram_before if (vram_peak and vram_before) else None,
            })

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT!")
            results.append({"ctx": ctx, "src": src_tokens, "comp": 0, "time": -1})
        finally:
            os.unlink(bin_path)

    # Summary table
    print("\n" + "=" * 100)
    print(f"{'Context':>10} | {'Src':>8} | {'Comp':>8} | {'Time (s)':>9} | {'Tok/s':>8} | {'VRAM (MB)':>12} | {'VRAM Delta':>12}")
    print("-" * 100)
    for r in results:
        tps = r["src"] / r["time"] if r["time"] > 0 else 0
        vram_str = f"{r['vram_peak']}" if r['vram_peak'] else "N/A"
        delta_str = f"{r['vram_delta']}" if r['vram_delta'] is not None else "N/A"
        print(f"{r['ctx']:>10} | {r['src']:>8} | {r['comp']:>8} | {r['time']:>9.2f} | {tps:>8.1f} | {vram_str:>12} | {delta_str:>12}")
    print("=" * 100)


if __name__ == "__main__":
    run()
