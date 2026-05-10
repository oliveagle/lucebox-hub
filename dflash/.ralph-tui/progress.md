# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

### Sampler Config Resolution Pattern
In `generate_internal()`, resolve sampling config with proper fallback:
```cpp
SamplerCfg effective_sampler = sampler;
if (effective_sampler.temp == 0.0f && effective_sampler.top_p == 1.0f &&
    effective_sampler.top_k == 0 && effective_sampler.rep_pen == 1.0f) {
    effective_sampler = sampler_cfg_;  // Use server defaults
}
if (effective_sampler.seed == 0) {
    effective_sampler.seed = std::random_device{}();
}
rng_.seed(effective_sampler.seed);  // Seed RNG for reproducibility
```

### Greedy vs Sampling Decision
```cpp
if (sampler.temp > 0.0f) {
    last_tok = sample_logits(logits, vocab, sampler, token_history_, rng_);
} else {
    last_tok = argmax_f32(logits, vocab);
}
```

### Token History for Repetition Penalty
```cpp
for (int i = 0; i < commit_n; i++) token_history_.push_back(replay_tok[i]);
if ((int)token_history_.size() > sampler.rep_window) {
    token_history_.erase(token_history_.begin(),
                         token_history_.end() - sampler.rep_window);
}
```

### Handler Sampling Parameter Extraction
Extract all sampling params from `ChatRequest` or `AnthropicMessagesRequest`:
```cpp
SamplerCfg req_sampler;
if (chat_req.temperature.has_value()) req_sampler.temp = *chat_req.temperature;
if (chat_req.top_p.has_value()) req_sampler.top_p = *chat_req.top_p;
if (chat_req.top_k.has_value()) req_sampler.top_k = *chat_req.top_k;
if (chat_req.repetition_penalty.has_value()) req_sampler.rep_pen = *chat_req.repetition_penalty;
if (chat_req.seed.has_value()) req_sampler.seed = *chat_req.seed;
generate_internal(prompt_tokens, max_tokens, on_token, has_tools, req_sampler);
```

---


## [2026-05-11] - STORY-008 - 高级功能

### What was implemented
1. **Sampling parameter integration** - Full integration of sampling parameters from API requests through to the decode loop
2. **Sampler configuration** - Added `SamplerCfg sampler_cfg_` as a class member to hold resolved sampling config
3. **Request-to-sampler mapping** - All handlers now extract and pass sampling parameters (temp, top_p, top_k, repetition_penalty, seed)
4. **Actual sampling in decode loop** - `run_dflash_decode` now uses `sample_logits()` when `temp > 0`
5. **RNG seeding** - Proper RNG seeding with request-specific seeds for reproducibility
6. **Token history tracking** - Added `token_history_` for repetition penalty support

### Files changed
- `src/server/dflash_server.h`:
  - Added `SamplerCfg sampler_cfg_` member variable
  - Moved `rng_` and `token_history_` to Sampling state section
  - Added `seed` field to `AnthropicMessagesRequest`
  - Updated `generate_internal` and `run_dflash_decode` signatures

- `src/server/dflash_server.cpp`:
  - Updated `generate_internal()` to resolve and apply sampling config
  - Updated `run_dflash_decode()` to use `sample_logits()` when temp > 0
  - Updated `handle_chat_completions()` to extract and pass sampling params
  - Updated `handle_chat_completions_stream()` to extract and pass sampling params
  - Updated `handle_anthropic_messages_impl()` to extract and pass sampling params
  - Updated `handle_anthropic_messages_stream_impl()` to extract and pass sampling params
  - Added seed parsing in `parse_chat_request()` and `parse_anthropic_messages_request()`
  - Added RNG initialization in `initialize()`

### Learnings:

**Patterns discovered:**
- **Sampler config resolution pattern**: The `generate_internal()` function resolves the effective sampling config by:
  1. Using the request-provided `SamplerCfg` if it has non-default values
  2. Falling back to server defaults (`sampler_cfg_`) if request params are all defaults
  3. Auto-generating a seed if none provided
  4. Storing the resolved config in `sampler_cfg_` class member for access by `run_dflash_decode()`
- **Token history for repetition penalty**: The `token_history_` vector is maintained across the generation loop and trimmed to `rep_window` size to support repetition penalty
- **RNG per-request seeding**: The `rng_` is re-seeded at the start of each `generate_internal()` call with the resolved seed to ensure reproducibility

**Gotchas encountered:**
- **Sampler signature propagation**: The `SamplerCfg` parameter had to be added through the entire call chain:
  - `generate_internal()` → `run_dflash_decode()`
  - All handlers (`handle_chat_completions`, `handle_chat_completions_stream`, etc.)
- **Header vs implementation mismatch**: The `run_dflash_decode()` function signature in the header had a default parameter `= {}` which needed to be removed
- **Missing seed field**: `AnthropicMessagesRequest` didn't have a `seed` field; had to add it as an OpenAI-compatible extension
- **Greedy vs sampling mode**: When `temp == 0`, the code uses `argmax_f32()` for greedy sampling; when `temp > 0`, it uses `sample_logits()` for proper sampling
- **Replay step sampling**: The sampling happens AFTER the replay step, using the last token's logits from `target_sg_.logits`

**Code patterns for reuse:**
```cpp
// Sampler config resolution pattern
SamplerCfg effective_sampler = sampler;
if (effective_sampler.temp == 0.0f && effective_sampler.top_p == 1.0f &&
    effective_sampler.top_k == 0 && effective_sampler.rep_pen == 1.0f) {
    effective_sampler = sampler_cfg_;  // Use server defaults
}
if (effective_sampler.seed == 0) {
    effective_sampler.seed = std::random_device{}();
}
sampler_cfg_ = effective_sampler;
rng_.seed(effective_sampler.seed);

// Greedy vs sampling decision
if (sampler.temp > 0.0f) {
    last_tok = sample_logits(last_logits.data(), vocab, sampler, token_history_, rng_);
} else {
    last_tok = argmax_f32(last_logits.data(), vocab);
}

// Token history management
for (int i = 0; i < commit_n; i++) {
    token_history_.push_back(replay_tok[i]);
}
if ((int)token_history_.size() > sampler.rep_window) {
    token_history_.erase(token_history_.begin(),
                         token_history_.end() - sampler.rep_window);
}
```

---
