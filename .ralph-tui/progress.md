# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

- **Qwen3.6-27B TriAttention stats**: Pre-built at `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt` (1.6MB). Covers layers 3-63 (every 4th layer), 48 heads each = 768 heads total. `head_dim=256`, `dtype=bfloat16`.

---

## 2026-05-18 - lucebox-hub-gfx1151-z0k.1
- Verified Qwen3.6-27B TriAttention stats file already exists and is valid
- File: `submodules/triattention/triattention/vllm/stats/qwen3_6_27b_stats.pt` (1.6MB, > 1MB threshold)
- Contains 768 head entries across layers 3-63 (full_attention layers only)
- Successfully loads via torch.load with metadata including head_dim=256, dtype=bfloat16, rope_style=half
- **Environment constraints discovered:**
  - GPU is Tesla GV100 (CC 7.0), driver 535.288.01 supports CUDA 12.2
  - venv PyTorch 2.11+cu130 incompatible with driver (CUDA 13.0 > driver 12.2)
  - lmdeploy venv PyTorch 2.10+cu128 works with GPU
  - AWQ models require gptqmodel, which is incompatible with PyTorch 2.10 due to Autotuner API changes
  - PyTorch 2.11 compiled without CC 7.0 support in cu128 builds
  - **Conclusion**: Pre-built stats file was necessary because on-demand calibration is blocked by env/driver constraints on this hardware
- **Learnings:**
  - Qwen3.6-27B has 16 full_attention layers out of 64 total (every 4th layer: 3,7,11,...,63)
  - The stats file uses only full_attention layers, consistent with Qwen3.6's hybrid attention+SSM architecture
  - Always check for existing pre-built stats before running calibration
---

