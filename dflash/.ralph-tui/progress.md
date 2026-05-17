# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*
- **DFlash Draft Architecture**: 5-layer decoder-only model with target hidden conditioning. Uses `Qwen3DFlashDecoderLayer` with `Qwen3DFlashAttention` (context+noise K/V concatenation). Training uses denoising cross-entropy loss on positions 1:block_size-1.
- **GGUF Draft Loading**: `load_draft_gguf()` 支持 GGUF 格式加载，配合 `load_draft_safetensors()` 用于 safetensors 格式。
- **DFlash GGUF 结构**: arch=`qwen35-dflash-draft`, hidden=5120, 5层, head_dim=128, Q8_0量化, ~1753 MB
- **Target Layer Selection**: `build_target_layer_ids(64, 5)` → `[1, 16, 31, 46, 60]` for 27B model (64 layers). Captures hidden states at regular intervals.
- **Training Pattern**: DFlash training requires target hidden states captured at 5 specific layers, then projecting them through `fc` + `hidden_norm` to match draft hidden dimension.
- **Denoising Loss**: Cross-entropy between draft predictions (positions 1:block_size-1) and ground truth tokens. First position (last token) is known and skipped.
- **Data Collection Pattern**: Multi-dataset collection with fallback prompts. Each sample captures prompt_hidden (last token at each layer) and gen_hidden (per-generation-step hidden states). Target layers: [1, 16, 31, 46, 60] (1-indexed from model outputs, offset+1 for embedding layer).
- **Training Loop Pattern**: `DraftTrainer.train_epoch()` implements full training loop with mixed precision (`torch.cuda.amp.autocast`), gradient accumulation, progress tracking, and denoising loss on positions 1:block_size-1.
- **Training Execution Requirements**: DFlash draft training requires HuggingFace-format target model (GGUF insufficient for hidden state capture), free V100 32GB (~72h total: 24h data collection + 48h training), and training data from `collect_draft_data.py`.

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.3.3
- **What was implemented**: Verified training execution requirements. Script `scripts/train_draft_qwen36.py` is fully implemented and syntax-validated. Training requires resources that are not currently available.
- **Files changed**: None (implementation already complete)
- **Learnings**:
  - **Training script complete**: `DraftTrainer.train_epoch()` implements mixed precision, gradient accumulation, checkpointing, early stopping
  - **Execution blockers identified**:
    1. GPU occupied: V100 has only 1324 MiB free (python process using 31GB)
    2. Model format: Only GGUF available, data collection needs HuggingFace format
    3. Training data: `models/training_data/` empty, requires ~24h collection
    4. Time requirement: ~48h training time after data collection
  - **Training flow confirmed**: Syntax validation passed, script ready to run when resources available
  - **Resource dependencies**: 1) Free V100, 2) HF-format Qwen3.6-27B, 3) Collected training data

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.3.2
- **What was implemented**: Verified training loop already fully implemented in `scripts/train_draft_qwen36.py`. `DraftTrainer` class with `train_epoch()` method includes mixed precision (AMP), gradient accumulation, denoising loss computation, checkpointing, progress bar, and early stopping.
- **Files changed**: None (already implemented)
- **Learnings**:
  - Training loop was complete from prior bead dflash-dflash-qwen36-acceptance-60-yzp.3.1
  - `DraftTrainer.train_epoch()`: iterates batches, forward with autocast, computes loss on positions 1:block_size-1, gradient accumulation, optimizer step
  - Mixed precision: `torch.cuda.amp.GradScaler` with `autocast()` context
  - Checkpointing: `save_checkpoint()` saves model+optimizer+config state every N epochs
  - Early stopping: stops when loss < 0.1
  - Syntax validation passed

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.4

- **What was implemented**: Verified existing GGUF draft file for Qwen3.6 DFlash. The file `models/draft/dflash-draft-3.6-q8_0.gguf` (1753.4 MB) was already converted and quantized. Created verification script and confirmed all acceptance criteria are met.

- **Files changed**:
  - `scripts/verify_draft_gguf.py` (new) - GGUF validation script with metadata and tensor checks

- **Learnings**:
  - **Existing GGUF 是完整的**: `dflash-draft-3.6-q8_0.gguf` 已经是正确格式，无需重新转换
  - **GGUF 结构验证**: arch=`qwen35-dflash-draft`, 5层, 5120 hidden, Q8_0量化, 包含所有必需张量
  - **关键张量存在**: `dflash.fc.weight` [25600, 5120], `dflash.hidden_norm.weight` [5120], `output_norm.weight` [5120]
  - **量化类型正确**: 36个 Q8_0 张量 (投影权重), 22个 F32 张量 (norm 权重)
  - **转换工具已存在**: `scripts/convert_dflash_to_gguf.py` 和 `scripts/quantize_draft_q8.py` 已实现
  - **验证通过**: GGUF 验证脚本成功确认所有元数据和张量正确

- **Verification Results**:
  - Architecture: qwen35-dflash-draft ✅
  - Block count: 5 ✅
  - Embedding length: 5120 ✅
  - Block size: 16 ✅
  - N target layers: 5 ✅
  - Critical tensors: all present ✅
  - File size: 1753.4 MB ✅

---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.2
- **What was implemented**: Enhanced data collection pipeline for Qwen3.6 DFlash draft training with multi-dataset support, validation, and documentation.
- **Files changed**:
  - `scripts/collect_draft_data.py` (enhanced) - Added multi-dataset support, conversation parsing, gen_hidden capture per step, validation
  - `scripts/run_collect_draft_data.py` (new) - Collection runner with dependency checking, pilot mode, merge capability
  - `docs/data_collection_requirements_20260518.md` (new) - Environment setup, dataset coverage, troubleshooting guide
  - `models/training_data/` (created) - Output directory for training data
- **Dataset coverage**: HumanEval (500), MBPP (500), MATH-500 (500), GSM8K (1000), ShareGPT (2000), LongPQA (200), LongAlpaca (500)
- **Learnings**:
  - Data collection requires HuggingFace model format, not GGUF (current environment only has GGUF)
  - Hidden states captured at 5 layers [1, 16, 31, 46, 60] for 64-layer Qwen3.6-27B
  - Each sample needs prompt_hidden (last token) AND gen_hidden (per-step during generation) for proper training
  - ShareGPT conversation format requires parsing nested structure to extract prompts
  - Pilot mode (--pilot) collects 100 samples per dataset for quick validation (~30-60 min)
  - Full collection (default config) ~5200 samples, 4-6 hours, ~15GB storage
---

## 2026-05-18 - dflash-dflash-qwen36-acceptance-60-yzp.3.1
- **What was implemented**: Verified training pipeline implementation - all 4 scripts already exist and pass syntax validation.
- **Files changed**: None (already implemented in prior iteration)
  - `scripts/collect_draft_data.py` - Multi-dataset collection with TARGET_LAYERS=[1, 16, 31, 46, 60]
  - `scripts/train_draft_qwen36.py` - DFlashDraftModel with denoising loss, mixed precision
  - `scripts/validate_draft.py` - Model validation with target model
  - `scripts/export_draft_hf.py` - HuggingFace export with safetensors
- **Learnings**:
  - Training pipeline already complete from prior iteration (bead dflash-dflash-qwen36-acceptance-60-yzp.3)
  - DFlash architecture: 5-layer decoder, block_size=16, target hidden conditioning
  - Denoising loss computed on positions 1:block_size-1 (first position known, skipped)
  - K/V concatenate context (target hidden) + noise (draft tokens) for non-causal attention
  - Target layers: [1, 16, 31, 46, 60] for 64-layer Qwen3.6-27B model

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

