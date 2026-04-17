# Contributing to Luce DFlash

This is a small research codebase with a clear scope (Qwen3.5-27B + z-lab DFlash draft, single RTX 3090). Plenty of room for contributors to add quality-of-life features and push performance further without needing to touch the CUDA kernels.

Below is a sorted list of concrete tasks with effort and expected impact. Pick one, open an issue referencing it, and send a PR.

## Good first issues (few hours to a day)

### 1. Streaming tokens from `test_dflash`
Right now `test_dflash` writes the full output `.bin` at the end. Add an optional `--stream-fd <N>` flag that writes each committed token id as little-endian `int32` to the given file descriptor as soon as it is accepted. Python side can read-while-running and print incrementally.

- Files: `test/test_dflash.cpp` (decode loop), `examples/chat.py` (consume stream)
- Benefits: chat feels interactive, shows tokens/s live

### 2. Daemon mode: keep model resident across turns
The chat REPL reloads the 15 GB target every turn (~10 s). Add a daemon mode to `test_dflash` (or split to a new `examples/chat_daemon.cpp`) that:

- Reads `<n_prompt><prompt_tokens><n_gen>\n` on stdin
- Writes `<token_id>\n` per committed token on stdout
- Keeps KV cache + target_feat across messages so multi-turn does not re-prefill the full conversation

- Files: new `examples/chat_daemon.cpp`, modify `examples/chat.py` to spawn it once
- Benefits: first-token latency drops to tens of ms

### 3. Temperature / top-k / top-p sampling
`test/test_dflash.cpp` uses `argmax` for both draft and verify. Add an optional sampler in the verify path that picks from top-k (temperature-scaled) tokens when the draft proposes a token that falls within the sampler's acceptance region — this is how the DFlash paper handles non-greedy decoding. Draft stays greedy.

- Files: `test/test_dflash.cpp` (verify path, `argmax` call sites)
- Benefits: creative writing / tool-use prompts stop being deterministic

### 4. CLI flag for `DFLASH27B_PREFILL_UBATCH` / `DFLASH27B_KV_Q4`
Right now these are env vars. Promote to flags in `test_dflash` + surface them via the Python wrappers so `--long-context` or `--kv-q4` "just works".

- Files: `test/test_dflash.cpp` arg parse, `scripts/run.py`, `examples/chat.py`

### 5. OpenAI-compatible HTTP server
Thin FastAPI wrapper over daemon mode exposing `/v1/chat/completions`. Talks the same streaming protocol as OpenAI's API. Enables Open WebUI / LM Studio / Cline / any existing chat client to point at Luce DFlash.

- Files: new `scripts/server.py` (depends on daemon mode from #2)
- Benefits: drop-in replacement for local chat clients

## Medium (a weekend to a week)

### 6. Q5_K_M / Q6_K target support
Only Q4_K_M is currently tested. Q5/Q6 likely recover the AL gap to the BF16 paper numbers at the cost of VRAM headroom. `gguf_target_loader.cpp` already reads Q4_K block layouts; extend to Q5_K / Q6_K (llama.cpp has reference code in `ggml-cuda`).

- Files: `src/gguf_target_loader.cpp`, target graph tensor type dispatch

### 7. CUDA graph capture for verify step
Capture the verify forward pass once, replay it each step. Saves ~3 ms per step (ggml graph build + dispatch). Straightforward to apply now that persistent gallocr is landed.

- Files: `test/test_dflash.cpp`
- Expected: +5-8 % tok/s

### 8. Adaptive tree budget
When the chain prefix of the draft is unanimous (top-1 probability > 0.9 for all 16 positions), skip the tree entirely and do a cheap chain verify. When draft is uncertain, fall back to full budget=22 tree.

- Files: `test/test_dflash.cpp` (DDTree builder call site)
- Expected: +3-5 % tok/s at no AL cost

## Hard (weeks of work, real perf wins)

### 9. AWQ INT4 target via ggml-cuda tensor cores
Q4_K_M today uses a hand-tuned matmul kernel that does not hit the RTX 3090's INT4 tensor cores. Porting to AWQ INT4 format + using `cutlass` / `triton` tensor-core gemms could cut verify time by ~30 %.

- Files: new kernel module under `deps/llama.cpp/ggml/src/ggml-cuda/` + loader changes
- Expected: +25-30 % mean tok/s

### 10. Megakernel verify path
Port the [Luce Megakernel](https://github.com/Luce-Org/luce-megakernel) approach to the 27B verify forward. Hybrid (attention + Gated DeltaNet) is harder than the 0.8B pure-DeltaNet case because SSM state updates need grid sync between layers, but the kernel-launch saving is ~100 ops per verify step.

- Files: new `deps/llama.cpp/ggml/src/ggml-cuda/qwen35_megaverify.cu`
- Expected: +20-40 % mean tok/s

### 11. Full llama.cpp integration
Register `qwen35-dflash-draft` as a first-class arch in upstream llama.cpp, add `llama-speculative-dflash.cpp`, wire flags into `common_params`. Once done, `llama-cli` and `llama-server` pick up DFlash for free — streaming, chat templates, OpenAI API, multi-user all come along.

- Files: upstream `deps/llama.cpp/src/llama-*`, new `tools/speculative-dflash/`
- Expected: production-grade UX, same tok/s as the standalone

## Development workflow

```bash
# Build + bench
cmake -B build -S . -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_dflash test_generate -j
python3 scripts/bench_llm.py

# Quick interactive test
python3 scripts/run.py --prompt "def fibonacci(n):"

# Multi-turn chat
python3 examples/chat.py
```

All benchmark numbers quoted in `RESULTS.md` are deterministic (greedy decode, seed=42 on dataset selection). Any PR touching the decode path should include a `bench_llm.py` run showing no regression on `HumanEval`.

## Style

- C++17, no exceptions in hot paths, no dynamic allocation in the decode loop.
- Prefer `static` / anonymous-namespace helpers over new public API symbols.
- CUDA code lives in `.cu` files. Prefer adding new kernels under `deps/llama.cpp/ggml/src/ggml-cuda/` (ggml's existing module layout) over a second CUDA dispatch mechanism.
- Python: standard library + `transformers` + `datasets`. Avoid adding heavy deps to `scripts/`.
