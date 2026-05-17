# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it is included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### vLLM Plugin Pattern

TriAttention uses the `vllm.general_plugins` entry point for automatic plugin discovery:

```python
# setup.py
entry_points={
    "vllm.general_plugins": [
        "triattention = triattention.vllm.plugin:register_triattention_backend",
    ],
}
```

When the package is installed via `pip install -e`, vLLM automatically discovers and loads the plugin on startup. The plugin function (`register_triattention_backend`) installs monkeypatches on the scheduler and worker.

**Key insight**: No code changes are needed in the user's application — the plugin modifies vLLM's behavior transparently via runtime monkeypatching.

### Environment Variable Naming Conventions

TriAttention uses `TRIATTN_RUNTIME_*` prefix for V2 runtime configuration, with legacy support for `TRIATTENTION_*` (without `_RUNTIME`). The plugin bridges legacy env vars to runtime vars automatically.

**When creating new integrations**:
- Use versioned prefixes (e.g., `*_RUNTIME_*`) to allow for future API evolution
- Provide bridge logic for backward compatibility
- Document the migration path clearly

### Calibration Data Format

TriAttention calibration produces `.pt` files with this structure:

```python
{
    "metadata": {
        "num_traces": int,
        "head_dim": int,
        "dtype": str,
        "rope_style": str,  # "half" or "interleaved"
        "rope_type": str,   # "default", "linear", etc.
        "sampled_heads": List[List[int]],  # [[layer, head], ...]
    },
    "stats": {
        f"layer{L:02d}_head{H:02d}": {
            "q_mean_real": Tensor,
            "q_mean_imag": Tensor,
            "q_abs_mean": Tensor,
        },
    }
}
```

The stats file is model-architecture specific but not weight-specific — one stats file works for all checkpoints of the same model family.

---

## 2026-05-18 - lucebox-hub-gfx1151-1oe

### What was implemented

TriAttention KV cache compression integration:
- **M3: vLLM support** — Created `scripts/run_vllm_with_triattention.sh` launcher script that sets required environment variables and invokes vLLM with correct flags (`--enforce-eager`, `--enable-prefix-caching false`)
- **M4: Calibration tool** — Created `scripts/calibrate_model.sh` wrapper for the triattention calibration script with sensible defaults
- **Documentation** — Added TriAttention section to README.md with quick start, configuration reference, and performance benchmarks

### Files changed

- `README.md` — Added TriAttention section (quick start, config, performance)
- `scripts/run_vllm_with_triattention.sh` — vLLM server launcher with TriAttention
- `scripts/calibrate_model.sh` — Calibration wrapper script

### Learnings

**vLLM plugin auto-discovery**: TriAttention registers itself via `vllm.general_plugins` entry point. Simply installing with `pip install -e submodules/triattention` enables the plugin — no code changes needed in the user's application.

**Required vLLM flags for TriAttention**:
- `--enforce-eager`: TriAttention's runtime hooks are not compatible with CUDA graphs
- `--enable-prefix-caching false`: Prefix caching interferes with KV compaction (compressed cache entries can be incorrectly reused)

**Submodule already initialized**: The triattention submodule was already registered in `.gitmodules` and initialized (M1 was complete). No `git submodule update` needed.

**Calibration is architecture-specific**: Frequency stats must be generated per model architecture (Qwen3, DeepSeek-R1, etc.) but work across all checkpoints of that family. The calibration script runs a single forward pass on text input and computes per-head frequency statistics.

**Community llama.cpp port exists**: The PRD mentioned M2 (llama.cpp + TriAttention), but this is handled by a community project ([triattention-ggml](https://github.com/domvox/triattention-ggml)) rather than the official TriAttention repo. No integration work needed in lucebox-hub for this path.

**M5 (verification testing) deferred**: End-to-end benchmark verification requires:
- A calibrated stats file for the target model
- Running AIME25/MATH-500 benchmarks
- Measuring throughput and accuracy

This should be done as a separate validation task after the integration is deployed.

---
