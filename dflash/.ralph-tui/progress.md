# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*
- **DFlash Draft Architecture**: 5-layer decoder-only model with target hidden conditioning. Uses `Qwen3DFlashDecoderLayer` with `Qwen3DFlashAttention` (context+noise K/V concatenation). Training uses denoising cross-entropy loss on positions 1:block_size-1.
- **Target Layer Selection**: `build_target_layer_ids(64, 5)` → `[1, 16, 31, 46, 60]` for 27B model (64 layers). Captures hidden states at regular intervals.
- **Training Pattern**: DFlash training requires target hidden states captured at specific layers, then projecting them through `fc` + `hidden_norm` to match draft hidden dimension.
- **Denoising Loss**: Cross-entropy between draft predictions (positions 1:block_size-1) and ground truth tokens. First position (last token) is known and skipped.

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.3
- **What was implemented**: Created complete training pipeline for Qwen3.6 DFlash draft model
- **Files changed**: 
  - `scripts/collect_draft_data.py` (new) - Data collection script for target hidden states
  - `scripts/train_draft_qwen36.py` (new) - Training script with mixed precision, gradient accumulation, denoising loss
  - `scripts/validate_draft.py` (new) - Model validation script
  - `scripts/export_draft_hf.py` (new) - HuggingFace format export script
- **Model specs**: 5 layers, hidden=5120, 32 heads, 8 KV heads, head_dim=128, block_size=16, ~1.8GB BF16
- **Learnings**:
  - DFlash draft uses non-causal denoising attention (is_causal=False)
  - K/V concatenate context (target hidden) and noise (draft tokens) for attention
  - Loss computed on positions 1:block_size-1, skipping first known position
  - Training requires capturing target hidden states at 5 specific layers [1, 16, 31, 46, 60]
---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.5
- **What was implemented**: Ran V100 baseline HumanEval benchmark (10 prompts, 128 output tokens). Results: MEAN AL=5.54, Accept Rate=35.6%, Decode Speed=39.98 tok/s. Current draft does NOT meet targets (AL>=8.0, AR>=60%, Speed>=60 tok/s).
- **Files changed**: `.ralph-tui/progress.md` (updated), no code changes
- **Learnings**:
  - Baseline V100 (May 2026): AL 5.54, 35.6%, 39.98 tok/s (n_gen=128, 10 HumanEval prompts)
  - Previous reported baseline: AL 6.05, 37.8%, 38.81 tok/s (similar range, n_gen likely different)
  - Tokenizer download requires HF mirror connectivity - use cached prompts with --skip-tokenize
  - Acceptance rate improvement requires newly trained draft on Qwen3.6 data (epic closed with training plan)
  - No 3090 resource available - documented reference data only

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.1
- **What was implemented**: Completed training requirements analysis for Qwen3.6 DFlash draft. All 4 child tasks closed.
- **Files changed**: No new files - all artifacts already existed from prior iterations
- **Verification**: 
  - Training plan: `docs/qwen36_draft_training_plan_20260518.md` (complete)
  - Training scripts: `scripts/collect_draft_data.py`, `scripts/train_draft_qwen36.py` (implemented)
  - Architecture analysis: 5-layer decoder, block_size=16, target hidden states at layers [1, 16, 31, 46, 60]
  - Qwen3.6 vs 3.5: SWA layers introduced, hidden distribution mismatch requires fresh training
  - Data requirements: 10K+ samples, diverse domains (code, math, conversation, long context)
  - GPU resources: V100 32GB, ~88h total (24h data collection + 48h training + 16h testing)
- **Learnings**:
  - DFlash draft is 5-layer non-causal transformer with target hidden conditioning
  - Loss is cross-entropy on positions 1:block_size-1 (first position skipped as known)
  - Qwen3.6 SWA layers fundamentally change attention patterns vs Qwen3.5 full attention
  - Draft trained on Qwen3.5 data won't transfer due to hidden state distribution mismatch
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

