# DFlash + TriAttention Integration Status

**Date**: 2026-05-18
**Task**: lucebox-hub-gfx1151-xlc
**Status**: Phase 1 Complete (Foundation) | Phase 2-3 TODO (Graph Integration)

---

## Overview

This document tracks the integration of **TriAttention KV compression** into **DFlash speculative decoding**. TriAttention provides 10.7× KV memory reduction and 2.5× throughput boost on AIME25 through frequency-domain scoring.

---

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DFlash + TriAttention                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐      ┌──────────────────┐      ┌────────────┐ │
│  │   Daemon    │─────▶│  TriAttention    │─────▶│ Stats File │ │
│  │  Init       │      │  C Library       │      │ (.bin)     │ │
│  └─────────────┘      └──────────────────┘      └────────────┘ │
│                                  │                              │
│                                  ▼                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              TargetCache (per-request state)            │   │
│  │  ┌─────────────────┐  ┌─────────────────────────────┐  │   │
│  │  │ tria_state      │  │ tria_k_pre_rope (TODO)       │  │   │
│  │  │ - stats_ptr     │  │ [fc, max_ctx, n_head_kv]     │  │   │
│  │  │ - enabled       │  └─────────────────────────────┘  │   │
│  │  │ - kv_budget     │                                    │   │
│  │  │ - divide_length │                                    │   │
│  │  └─────────────────┘                                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                  │                              │
│                                  ▼                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │          build_full_attn_block() (TODO)                 │   │
│  │  1. Capture pre-RoPE K                                  │   │
│  │  2. Call tria_score_kv_head() for each KV head         │   │
│  │  3. Select top-K based on scores                        │   │
│  │  4. Compact KV cache in-place                           │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase Status

### ✅ Phase 1: Foundation (COMPLETE)

**Components Implemented:**

1. **TriAttention C Library** (`dflash/deps/llama.cpp/triattention/`)
   - `tria_load()` - Load TRIA v2 binary stats
   - `tria_free()` - Free stats
   - `tria_score_kv_head()` - Score keys for one KV head

2. **C++ Wrapper** (`dflash/src/triattention_runner.{h,cpp}`)
   - `TriAttentionState` - Runtime state management
   - `init_triattention_from_env()` - Environment variable initialization
   - `load_stats()` - Load stats from file
   - `should_compress()` - Check if compression should trigger
   - `layer_budget()` - Calculate per-layer KV budget

3. **Build Configuration** (`dflash/CMakeLists.txt`)
   - `DFLASH27B_TRIATTENTION` option
   - Links `libtriattention.a` when enabled
   - Sets `DFLASH27B_TRIATTENTION_ENABLED=1` compile flag

4. **Stats Files** (`dflash/deps/llama.cpp/triattention/stats/`)
   - `qwen3.5-27b.bin` (768.3K) - For Qwen3.5-27B
   - `qwen3-8b.bin` (1.1M) - For Qwen3-8B
   - `qwen3-1.7b.bin` (448.1K) - For Qwen3-1.7B

5. **TargetCache Integration** (`dflash/src/internal.h`)
   - Added `TriAttentionState tria_state` to `TargetCache`
   - Added `tria_k_pre_rope` placeholder for pre-RoPE K cache

6. **Daemon Initialization** (`dflash/src/qwen35/qwen35_daemon.cpp`)
   - Calls `init_triattention_from_env()` at daemon startup
   - Logs TriAttention enable/disable status

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `TRIATTN_STATS_PATH` | `dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin` | Stats file path |
| `TRIATTN_KV_BUDGET` | 2048 | Max tokens to retain per request |
| `TRIATTN_DIVIDE_LENGTH` | 128 | Compression trigger interval |
| `TRIATTN_WINDOW_SIZE` | 128 | Recent tokens always preserved |

---

### ⏳ Phase 2: Pre-RoPE K Capture (TODO)

**Challenge:** DFlash applies RoPE inline in `build_full_attn_block()`. TriAttention requires pre-RoPE K for frequency-domain scoring.

**Approach Options:**

1. **Split RoPE Operation** - Separate `ggml_rope_multi()` into K-rope and Q-rope, store pre-rope K
2. **Separate K Cache** - Maintain a parallel pre-RoPE K cache alongside the standard KV cache

**Files to Modify:**
- `dflash/src/qwen35/qwen35_target_graph.cpp::build_full_attn_block()`
- Add pre-RoPE K capture after Kcur projection
- Store in `cache.tria_k_pre_rope`

---

### ⏳ Phase 3: KV Scoring & Pruning (TODO)

**Challenge:** After scoring, need to compact the KV cache in-place while preserving position mapping.

**Implementation Steps:**

1. **Score KV Heads** - Call `tria_score_kv_head()` for each KV head
2. **Aggregate Scores** - Combine scores across GQA groups
3. **Select Top-K** - Choose indices to keep based on layer budget
4. **Compact KV** - Rewrite cache to keep only selected indices
5. **Update Positions** - Remap position IDs after compaction

**Files to Modify:**
- `dflash/src/qwen35/qwen35_target_graph.cpp::build_full_attn_block()`
- Add scoring logic after KV write
- Implement compaction using ggml ops

**Pseudo-code:**

```cpp
// In build_full_attn_block(), after KV write
if (cache.tria_state.enabled && cache.tria_state.should_compress(kv_start + n_tokens)) {
    // Capture pre-RoPE K (requires Phase 2)
    ggml_tensor * k_pre = /* pre-RoPE K */;

    // Score each KV head
    std::vector<float> scores(seq_len * n_head_kv);
    for (int h = 0; h < n_head_kv; h++) {
        tria_score_kv_head(
            cache.tria_state.stats_ptr,
            k_pre_real, k_pre_imag,
            key_positions,
            kv_start + n_tokens,
            seq_len,
            layer_idx,
            h,
            scores.data() + h * seq_len
        );
    }

    // Select top-K based on layer budget
    int budget = cache.tria_state.layer_budget(layer_idx, cache.tria_state.stats_ptr);
    std::vector<int> keep_indices = select_top_k(scores, budget);

    // Compact KV cache
    compact_kv_cache(cache_k, cache_v, keep_indices);

    cache.tria_state.mark_compressed(kv_start + n_tokens);
}
```

---

## Build Instructions

### Enable TriAttention at Build Time

```bash
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash
cmake -B build -S . -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DDFLASH27B_TRIATTENTION=ON
cmake --build build --target test_dflash -j
```

### Run with TriAttention

```bash
# Set stats path (optional, uses default if not set)
export TRIATTN_STATS_PATH="dflash/deps/llama.cpp/triattention/stats/qwen3.5-27b.bin"

# Set KV budget (optional, default 2048)
export TRIATTN_KV_BUDGET=2048

# Run DFlash daemon
./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/dflash-draft-3.6-q8_0.gguf
```

---

## Validation

### Acceptance Criteria

- [x] DFlash 可加载 TriAttention 统计文件 (Stats loading works)
- [ ] decode 阶段 KV cache 被压缩 (KV compression in decode loop - TODO)
- [ ] 精度损失可接受 (< 5%) (Accuracy validation - TODO)
- [ ] 吞吐量提升明显 (Throughput measurement - TODO)

### Testing Plan

1. **Stats Loading Test** - Verify stats file loads correctly
2. **Memory Reduction Test** - Measure KV memory before/after compression
3. **Accuracy Test** - Compare outputs with/without compression on HumanEval
4. **Throughput Test** - Measure tok/s improvement on long-context benchmarks

---

## Alternative: vLLM + TriAttention

Given the architectural challenges of C++ graph integration, the **recommended approach** is to use **vLLM + TriAttention** for KV compression separately:

```bash
# vLLM + TriAttention is already working
./scripts/run_vllm_with_triattention.sh Qwen/Qwen3-8B --max-model-len 32768
```

This path is fully functional and provides:
- 10.7× KV memory reduction
- 2.5× throughput boost on AIME25
- No accuracy loss

Use **DFlash** for speculative decoding speed and **vLLM + TriAttention** for long-context KV compression. Benchmark both paths to quantify trade-offs.

---

## References

- [TriAttention README](../submodules/triattention/README.md)
- [TriAttention vLLM Integration](../submodules/triattention/docs/vllm.md)
- [TriAttention Calibration Guide](../submodules/triattention/docs/calibration.md)
- [DFlash README](../dflash/README.md)
- [PRD: TriAttention Integration](../docs/prd_triattention_integration_20260518.md)
