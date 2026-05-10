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

### Type Definition Pattern
- `StepGraph`, `DraftFeatureMirror` are namespace-level structs in `dflash27b`, NOT nested inside `DFlashServer` class.
- When defining static helper functions that take these types, use `StepGraph&` NOT `DFlashServer::StepGraph&`.
- Class definitions used by member functions must be defined BEFORE the member function implementations in the .cpp file.

### Header File Pattern
- `qwen35_target_graph.h` does NOT exist - functions are declared in `internal.h`.
- Use `#include "internal.h"` for graph building functions (`build_qwen35_graph`, `snapshot_ssm_state`, `restore_ssm_state`, etc.).
- Use `#include "qwen3_drafter.h"` for PFlash drafter functions (`load_drafter`, `drafter_score_and_compress`, `free_drafter`).

### Macro Definition Pattern
- `IS_EOS_TOK(tok, w)` macro is defined in test files but NOT in headers - need to define it locally if needed.
- Definition: `#define IS_EOS_TOK(tok, w) ( ((w).eos_chat_id >= 0 && (tok) == (w).eos_chat_id) || ((w).eos_id >= 0 && (tok) == (w).eos_id) ) )`

### CMake Build Pattern
- HTTP server target requires `-DDFLASH27B_HTTP_SERVER=ON` to build `dflash_server` executable.
- Build command: `cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DDFLASH27B_HTTP_SERVER=ON && cmake --build build --target dflash_server`

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

## 2025-05-10 - STORY-003
- **What was implemented**: PFlash 压缩集成。实现了 PrefillCoordinator 压缩决策逻辑（支持 off/auto/always 模式），drafter 加载/释放函数，PFlash 压缩函数（调用 drafter_score_and_compress），集成到 generate_internal() 的 prefill 阶段。新增命令行参数：--prefill-mode, --prefill-threshold, --prefill-keep-ratio, --prefill-drafter。
- **Files changed**:
  - `src/server/dflash_server.h`: 添加 DrafterContext 前向声明，drafter_ctx_ 和 drafter_loaded_ 成员变量，PFlash 方法声明（load_drafter, free_drafter, compress_prefill）。
  - `src/server/dflash_server.cpp`: 添加 qwen3_drafter.h include，实现 PrefillCoordinator 类（压缩决策），实现 PFlash 方法（load_drafter, free_drafter, compress_prefill），在 generate_internal() 中集成 PFlash 压缩（park draft → compress → restore draft），添加 IS_EOS_TOK 宏定义，添加命令行参数解析，移除重复的 generate_internal 占位函数。
- **Learnings:**
  - `StepGraph` 是命名空间级别的结构体，不是 `DFlashServer` 的嵌套类，静态辅助函数应使用 `StepGraph&` 而非 `DFlashServer::StepGraph&`。
  - `PrefillCoordinator` 和 `PrefixCache` 类定义需要在使用它们的函数之前定义，避免前向声明问题。
  - `IS_EOS_TOK` 宏定义在 test_dflash.cpp 中，需要复制到 server 代码（检查 eos_id 和 eos_chat_id）。
  - CMake 需要 `-DDFLASH27B_HTTP_SERVER=ON` 才能编译 dflash_server 目标。
  - `qwen35_target_graph.h` 不存在，相关函数声明在 internal.h 中。
- **Result**: Build succeeds, executable at `./dflash_server` (1.6MB).

---
