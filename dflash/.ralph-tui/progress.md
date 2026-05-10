# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

### Model Loading Pattern
- All model loading functions (`load_target_gguf`, `load_draft_safetensors`, `load_draft_gguf`) require a `ggml_backend_t` parameter passed as the second argument, not embedded in the weights struct.
- Create the CUDA backend with `ggml_backend_cuda_init(0)` before loading models.
- Field names in `TargetWeights` and `DraftWeights` use `n_layer` (singular), not `n_layers`.
- `create_target_cache()` signature: `(const TargetWeights& w, int max_ctx, int max_verify_tokens, ggml_backend_t backend, TargetCache& out, bool prefill_only = false)`
- Use `ggml-cuda.h` header for `ggml_backend_cuda_init()` (not just `ggml-backend.h`).
- `Tokenizer` class is in the `dflash27b` namespace (same as server code), so no namespace alias needed.
- Use C++17-compatible string checks instead of C++23's `ends_with()` (e.g., `str.size() >= N && str.substr(str.size() - N) == suffix`).

---

## 2025-05-10 - STORY-001
- **What was implemented**: Removed placeholder Tokenizer and ChatTemplate classes from dflash_server.cpp, integrated real Tokenizer from src/server/tokenizer.h, implemented model loading in initialize() with proper CUDA backend creation, KV cache type resolution from environment variables, target/draft model loading, and KV cache creation.
- **Files changed**:
  - `src/server/dflash_server.cpp`: Removed placeholder Tokenizer class (18 lines), removed placeholder ChatTemplate class (11 lines), added includes for tokenizer.h, kv_quant.h, ggml.h, ggml-backend.h, ggml-cuda.h, implemented model loading in initialize() with CUDA backend, KV type resolution, target/draft loading, cache creation.
  - `src/server/dflash_server.h`: Removed ChatTemplate forward declaration and chat_template_ member variable, kept Tokenizer forward declaration, removed apply_chat_template helper declaration.
- **Learnings:**
  - Model loading APIs require explicit `ggml_backend_t` parameter, not implicit in weights struct.
  - Field names use `n_layer` (singular) not `n_layers`.
  - `create_target_cache()` takes `max_ctx`, `max_verify_tokens`, then `backend`, not `kv_k_type`/`kv_v_type` directly (those are set via env vars and stored in cache struct).
  - Need `ggml-cuda.h` header for CUDA backend initialization.
  - `ends_with()` is C++23, project uses C++17.
- **Result**: Build succeeds, executable at `build/dflash_server` (1.5MB).

---
