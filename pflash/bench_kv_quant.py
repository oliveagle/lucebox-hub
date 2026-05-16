#!/usr/bin/env python3
"""PFlash KV Quantization Benchmark for ROCm/gfx1151."""
import ctypes, struct, subprocess, os, time, tempfile
from transformers import AutoTokenizer
import threading

DRAFTER_GGUF = "/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3-0___6B-GGUF/Qwen3-0.6B-BF16.gguf"
DAEMON_BIN = "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/build/pflash_daemon"

TEST_CTX = [8192, 16384, 32768, 65536]
KEEP_RATIO = 0.05

# KV quantization options to test
KV_CONFIGS = [
    {"name": "F16 (default)", "k": "f16", "v": "f16"},
    {"name": "Q4_0 / Q8_0", "k": "q4_0", "v": "q8_0"},
    {"name": "Q8_0 / Q8_0", "k": "q8_0", "v": "q8_0"},
    {"name": "Q4_0 / Q4_0", "k": "q4_0", "v": "q4_0"},
    {"name": "TQ3_0 / Q8_0", "k": "tq3_0", "v": "q8_0"},
]

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


def run_kv_test():
    print("=" * 110)
    print("PFlash KV Quantization Benchmark — ROCm gfx1151")
    print("=" * 110)

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
    passage = (
        "The quick brown fox jumps over the lazy dog. "
        "In the beginning, the universe was created. "
        "This is a test of long context processing. "
        "Machine learning is transforming the world."
    )

    results = []

    for kv_config in KV_CONFIGS:
        print(f"\n{'='*100}")
        print(f"KV Config: {kv_config['name']} (K={kv_config['k']}, V={kv_config['v']})")
        print(f"{'='*100}")

        for ctx in TEST_CTX:
            # Generate prompt
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

                # Run daemon with KV env vars
                env = {
                    **os.environ,
                    "LD_LIBRARY_PATH": "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:" +
                                   "/home/oliveagle/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:" +
                                   os.environ.get("LD_LIBRARY_PATH", ""),
                    # Note: drafter currently forces TQ3=0 internally
                    # These env vars may not affect pflash compress
                    "DFLASH27B_KV_K": kv_config['k'],
                    "DFLASH27B_KV_V": kv_config['v'],
                }

                cmd = f"echo 'compress {keep_x1000} 8 32 13 {bin_path}' | {DAEMON_BIN} {DRAFTER_GGUF}"
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    text=True,
                    timeout=600,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )

                stop_sampling = True
                sampler.join(timeout=2)
                elapsed = time.time() - t0

                # Parse output
                comp_count = 0
                for line in proc.stdout.splitlines():
                    if "compress done" in line:
                        for p in line.split():
                            if p.startswith("out="):
                                try:
                                    comp_count = int(p.split("=")[1])
                                except:
                                    pass

                vram_peak = max(vram_samples) if vram_samples else None
                vram_delta = vram_peak - vram_before if (vram_peak and vram_before) else None

                print(f"  {ctx:>6} tok: {elapsed:>6.2f}s, VRAM peak: {vram_peak:>6} MB, delta: {vram_delta:>+6} MB, comp: {comp_count}")

                results.append({
                    "kv_name": kv_config['name'],
                    "kv_k": kv_config['k'],
                    "kv_v": kv_config['v'],
                    "ctx": ctx,
                    "src": src_tokens,
                    "comp": comp_count,
                    "time": elapsed,
                    "vram_peak": vram_peak,
                    "vram_delta": vram_delta,
                })

            except subprocess.TimeoutExpired:
                print(f"  {ctx:>6} tok: TIMEOUT!")
            finally:
                os.unlink(bin_path)

    # Summary table by KV config
    print("\n" + "=" * 110)
    print(f"{'KV Config':>20} | {'16K':>28} | {'32K':>28} | {'64K':>28}")
    print(f"{'':21}|{'VRAM':>8} {'Time':>8} {'Tok/s':>8}|{'VRAM':>8} {'Time':>8} {'Tok/s':>8}|{'VRAM':>8} {'Time':>8} {'Tok/s':>8}")
    print("-" * 110)

    for kv in KV_CONFIGS:
        row = f"{kv['name']:>20} |"
        for ctx in [16384, 32768, 65536]:
            r = next((x for x in results if x['kv_name'] == kv['name'] and x['ctx'] == ctx), None)
            if r and r['vram_peak']:
                tps = r['src'] / r['time'] if r['time'] > 0 else 0
                row += f" {r['vram_peak']:>6}MB {r['time']:>6.2f}s {tps:>6.0f} |"
            else:
                row += " " * 28 + "|"
        print(row)
    print("=" * 110)

    # VRAM savings summary
    print("\nVRAM Savings vs F16 (at 64K context):")
    f16_64k = next((x for x in results if x['kv_name'] == "F16 (default)" and x['ctx'] == 65536), None)
    if f16_64k and f16_64k['vram_peak']:
        for kv in KV_CONFIGS:
            if kv['name'] == "F16 (default)":
                continue
            r = next((x for x in results if x['kv_name'] == kv['name'] and x['ctx'] == 65536), None)
            if r and r['vram_delta']:
                saved = f16_64k['vram_peak'] - r['vram_peak']
                pct = (saved / f16_64k['vram_peak'] * 100) if f16_64k['vram_peak'] > 0 else 0
                print(f"  {kv['name']:>20}: saved {saved:>5} MB ({pct:>5.1f}%)")


if __name__ == "__main__":
    run_kv_test()
