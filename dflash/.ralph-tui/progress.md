# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-11] - STORY-002 - 核心生成循环 (Core Generation Loop)

**Status**: ✅ Complete

### What was implemented
The core generation loop was already fully implemented in `src/server/dflash_server.cpp`:

1. **`generate_internal()`** (lines 1124-1309)
   - Prefix cache lookup/restore
   - PFlash compression (on cache miss)
   - Prefill with layer feature capture
   - DDTree mode selection based on `config_.ddtree_budget`

2. **`run_dflash_decode()`** (lines 612-989)
   - Dual-path: DDTree mode (tree-based) and Chain mode (standard)
   - Noise block construction: `[last_tok, MASK*15]`
   - Draft forward → 16 candidate tokens
   - SSM state snapshot before verification
   - Batch target verify with causal mask
   - Greedy longest-prefix accept + bonus token
   - SSM state restore
   - Token replay
   - EOS detection

3. **DDTree support** (lines 142-346)
   - `extract_draft_topk()` - Top-K extraction for tree building
   - `build_ddtree()` - Best-first search tree builder
   - `follow_verified_tree()` - Walk verified tree for accepted path

### Files changed
No changes - the code was already complete.

### Learnings

**Patterns discovered**:
- **SSM state management**: DeltaNet layers require snapshotting SSM + conv state before target verification for rollback capability
- **Dual-path generation**: DDTree (tree-based) vs Chain (linear) modes selected via `config_.ddtree_budget > 0`
- **Graph reuse**: `StepGraph` structures (`target_sg_`, `draft_sg_`, `proj_sg_`) are reused across calls to avoid allocation overhead
- **Token callback pattern**: `on_token` callback enables both streaming (SSE) and non-streaming modes with same core logic

**Gotchas encountered**:
- None - the implementation was already correct and complete

---
