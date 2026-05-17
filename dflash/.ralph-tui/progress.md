# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [Date] - dflash-dflash-qwen36-acceptance-60-yzp

- **What was implemented**: Analyzed Qwen3.6 draft acceptance rate issue and created comprehensive training plan

- **Files changed**: 
  - `docs/qwen36_draft_training_plan_20260518.md` (new)
  - `.ralph-tui/progress.md` (updated)

- **Learnings**:
  - Qwen3.6 introduces SWA (Sliding Window Attention) which changes attention patterns vs Qwen3.5
  - Current drafts trained on Qwen3.5 data don't match Qwen3.6 hidden states distribution
  - DFlash draft is a 5-layer model that predicts 16 tokens per step using captured target hidden states from layers [10, 20, 30, 40, 50]
  - V100 (SM70) vs RTX 3090 (SM86) has 50% decode speed gap due to hardware, but V100 has HIGHER acceptance rate (37.8% vs 32.3%) on Qwen3.6
  - Z-lab has NOT yet released training scripts (they mention "coming soon" in README)
  - DIY training requires ~88h V100 compute (24h data collection + 48h training + 16h testing)
  - Acceptance rate 60%+ requires fresh training on Qwen3.6 target data, not conversion/port of Qwen3.5 draft

---

## [Date] - dflash-v100-flashprefill-optimization-v3i

- **What was implemented**: V100 FlashPrefill optimization with hybrid block-select + ggml flash_attn_sparse

- **Files changed**:
  - `src/flashprefill_q8.cpp` - Modified to use ggml_flash_attn_sparse
  - `src/pflash_ggml_adapter.cpp` - Added namespace and conditional compilation
  - `CMakeLists.txt` - Added adapter for sm_70 build
  - `docs/v100_flashprefill_optimization_final_report_20260517.md` (new)
  - `docs/v100_optimization_proposal_20260517.md` (updated)

- **Learnings**:
  - **Pattern: ggml_flash_attn_sparse registration** - Bypasses ggml op dispatch to call custom kernels directly
  - **Pattern: E2E vs Micro-benchmark** - Micro-benchmarks test individual kernels, E2E tests full pipeline; tail-score time includes multiple stages
  - **Gotcha: pflash_daemon input format** - Requires binary token files with format: `u32 count (LE) + count × int32 token IDs`
  - **Gotcha: ggml_cont necessity** - `ggml_permute` creates view, `ggml_cont` required for data contiguity
  - **Gotcha: Sparse kernel crash** - sparse_flash_forward_f16 kernel has memory issues at S >= 16384, bypassed via hybrid approach

---

