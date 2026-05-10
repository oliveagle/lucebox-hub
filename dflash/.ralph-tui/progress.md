# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

### llama.cpp Tokenizer Integration Pattern

When integrating llama.cpp's tokenizer into a C++ project:

1. **CMake Configuration**: Add llama.cpp as a subdirectory with minimal options:
   ```cmake
   set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
   set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
   set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
   set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
   set(LLAMA_STANDALONE OFF CACHE BOOL "" FORCE)
   add_subdirectory(deps/llama.cpp)
   ```

2. **Include Paths**: Must include both `deps/llama.cpp` and `deps/llama.cpp/include`:
   ```cmake
   target_include_directories(my_target PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/deps/llama.cpp/include
   )
   ```

3. **Link Libraries**: Must link against `llama` library:
   ```cmake
   target_link_libraries(my_target PRIVATE llama ggml ...)
   ```

4. **API Usage**:
   - Use `llama_model_load_from_file()` with `vocab_only=true` for tokenizer-only loading
   - Get vocab with `llama_model_get_vocab(model)`
   - Encode with `llama_tokenize(vocab, text, ...)`
   - Decode with `llama_token_to_piece(vocab, token, ...)`
   - Get special tokens with `llama_vocab_bos/eos/eot()`

5. **Context Creation**: For tokenizer-only use, create minimal context:
   ```cpp
   llama_context_params cparams = llama_context_default_params();
   cparams.n_ctx = 8192;  // Small context for tokenizer
   cparams.offload_kqv = false;
   cparams.embeddings = false;
   // Note: no_alloc field was removed in recent llama.cpp versions
   ```

---

## [Date] - STORY-001

**Date**: 2026-05-10

**Task**: Tokenizer Integration (STORY-001 - 核心功能)

**What was implemented**:
- Created complete tokenizer implementation using llama.cpp's API
- Updated `src/server/tokenizer.h` with proper interface
- Updated `src/server/tokenizer.cpp` with working implementation
- Created `src/server/tokenizer_parser.cpp` with helper utilities
- Updated `src/server/tokenizer_parser.h` to add helper functions
- Updated `CMakeLists.txt` to integrate llama.cpp library

**Files changed**:
- `src/server/tokenizer.h` - Rewritten with proper llama.cpp integration
- `src/server/tokenizer.cpp` - Implemented encode/decode using llama_tokenize and llama_token_to_piece
- `src/server/tokenizer_parser.h` - Added helper utilities for common patterns
- `src/server/tokenizer_parser.cpp` - Created with utility implementations
- `CMakeLists.txt` - Added llama.cpp subdirectory and linked llama library

**Learnings**:

1. **llama.cpp API Changes**: The current llama.cpp API has changed from older versions:
   - `llama_tokenize()` now takes `llama_vocab*` as first argument, not `llama_context*`
   - `llama_token_to_piece()` now takes `llama_vocab*`, not `llama_model*`
   - Special token functions moved: `llama_token_bos/eos/eot()` → `llama_vocab_bos/eos/eot()`
   - `llama_n_vocab()` → `llama_vocab_n_tokens()`
   - `llama_context_params::no_alloc` field was removed

2. **Getting Vocab from Model**: Use `llama_model_get_vocab(model)` to get vocab from model:
   ```cpp
   auto vocab = llama_model_get_vocab(model);
   llama_vocab_n_tokens(vocab);
   llama_vocab_bos(vocab);
   ```

3. **Tokenizer-Only Loading**: For tokenizer-only use, can use:
   ```cpp
   llama_model_params mparams = llama_model_default_params();
   mparams.n_gpu_layers = 0;  // CPU-only for tokenizer
   ```

4. **Include Path Requirements**: Both `deps/llama.cpp` and `deps/llama.cpp/include` are required because:
   - `deps/llama.cpp/include/llama.h` is the main header
   - Some internal headers are in `deps/llama.cpp/src`

5. **Build System Integration**: Adding llama.cpp as subdirectory (without `EXCLUDE_FROM_ALL`) ensures the `llama` library target is built and available for linking.

6. **Const Correctness**: `llama_free()` expects non-const `llama_context*`, so store as non-const pointer in the wrapper class.

---
