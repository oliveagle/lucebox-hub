# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## 2026-05-17 - dflash-v100-flashprefill-optimization-v3i.2.1
- **Bead**: dflash-v100-flashprefill-optimization-v3i.2.1 - 定位 block_select bug
- **What was implemented**: 验证 block_select_kernel (lines 598-664) 功能正常，定位真正的 bug 在 `sparse_flash_forward_f16` WMMA attention kernel
- **Files analyzed**:
  - `src/flashprefill_f16.cu` - block_select_kernel (598-685)
  - `test/bench_flashprefill_e2e.cpp` - E2E benchmark
  - `docs/block_select_investigation_20260517.md` - 已有调查结论
- **结论**: block_select 在 S=320..65536 范围内稳定运行，**真正的 crash 在 sparse_flash_forward kernel (S >= 16384 时触发)**
- **Learnings:**
  - **Pattern: block_select_kernel 使用 1 warp per (B,M,H)**
    - 32 个 thread，执行 warp reduce 和 warp ballot compact
    - 两 Pass: Pass 1 求 max score，Pass 2 predicate + compact
  - **Pattern: 边界条件检查**
    - Pass 1: `(n <= m) && (n < N)` - 防止越界读 score
    - Pass 2: 同样的 valid 检查，确保不读无效数据
  - **Gotcha: 错误归因**
    - 用户报告 "block_select 导致崩溃"，但实际 crash 在 `sparse_flash_forward_f16` (WMMA attention kernel)
    - 在定位 bug 时需要分别测试各 kernel，而非假设报告准确
  - **Gotcha: 大序列长度**
    - S >= 16384 时 sparse_flash_forward 触发 illegal memory access
    - block_select 本身在此范围无问题
---

