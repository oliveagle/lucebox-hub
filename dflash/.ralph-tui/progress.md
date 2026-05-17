# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-05-18 - dflash-v100-flashprefill-optimization-v3i
- What was implemented
  - Epic "V100 FlashPrefill 优化" 验收并关闭
  - 确认所有子任务已完成: 方向 A (ggml WMMA 集成)、方向 B (GEMM block_score)、方向 C (混合 attention)、E2E 测试、性能基准测试、最终报告
  - 65K tokens 性能: 10.17s (GEMM) / 12.09s (Scalar)，超越 < 20s 目标 50%
  - 8.0× 加速相比基线 81.48s
- Files changed
  - 无代码变更 (全部在 previous iteration 完成)
  - 验收 docs/v100_flashprefill_optimization_final_report_20260517.md
- **Learnings:**
  - 三个优化方向最终形成互补方案: 方向 A 集成 kernel、方向 B 提供 GEMM block_score、方向 C 混合 attention 是最终生效方案
  - 混合 attention 架构结合了 DFlash block-select 稀疏选择和 ggml flash_attn_sparse 密集计算
  - V100 (SM70) 受限于 Volta F16 WMMA，通过混合方案绕过原生 BF16/BSA 支持缺失
  - ggml_flash_attn_sparse 注册机制是集成自定义 kernel 到 ggml pipeline 的标准方式
  - pflash_daemon 需要 binary token 文件输入 (u32 count LE + int32 tokens)
  - E2E 性能测试 (pflash_daemon compress) 与 Micro-benchmark 测量维度不同，需区分
  - GEMM block_score 在长序列 (65K) 下提供 19% 加速，短序列 (4K) 几乎无加速

