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

## [Date] - STORY-002

**Date**: 2026-05-10

**Task**: Tool Call Parsing (STORY-002 - 高级功能)

**What was implemented**:
- Created `src/server/parsers.h` with tool call and reasoning parsing interfaces
- Created `src/server/parsers.cpp` with complete implementation of:
  - `parse_tool_calls()` - Parses Qwen3 tool call format (访问函数...结束访问)
  - `parse_reasoning()` - Parses Qwen3 thinking/reasoning tags (反复推理...结束推理)
  - `convert_param_value()` - Type coercion for tool parameters (string/int/float/bool/object/array/null)
  - `find_tool_properties()` - Helper to find tool schema by function name
  - `generate_call_id()` - Generates random call IDs for tool calls
- Updated `CMakeLists.txt` to add `src/server/parsers.cpp` to dflash_server build

**Files changed**:
- `src/server/parsers.h` - Created with tool call and reasoning parsing interfaces
- `src/server/parsers.cpp` - Created with parser implementations ported from vLLM
- `CMakeLists.txt` - Added parsers.cpp to dflash_server sources

**Learnings**:

1. **C++ std::regex doesn't have dotall mode**: Unlike Python's `re.DOTALL` or PCRE's `s` flag, C++ `std::regex` has no equivalent. To match across newlines, use `[\s\S]*?` instead of `.*?`. The `[\s\S]` character class matches any character including newlines.

2. **Avoid name collisions with `std::end`**: When using `std::sregex_iterator`, don't name the end iterator `end` as it can conflict with `std::end` in some contexts. Use `iter_end` or similar instead.

3. **Qwen3 tool call format**: The XML format uses Chinese tags:
   - Tool calls: `访问函数<function=NAME>...</function>结束访问`
   - Reasoning: `反复推理...结束推理`
   - The regex patterns need to handle UTF-8 encoded Chinese characters correctly.

4. **Type coercion for tool parameters**: Parameters extracted from XML are strings. The `convert_param_value()` function converts them to proper JSON types based on the tool's JSON schema:
   - `int`, `uint`, `long`, `short`, `unsigned` → integer
   - `num`, `float` → float (or int if whole number)
   - `boolean`, `bool` → boolean (true/false)
   - `string`, `str`, `text`, etc. → string
   - `object`, `array`, `dict`, `list` → JSON-parsed object/array
   - `null` → JSON null

5. **Tool call ID generation**: Call IDs follow OpenAI format: `call_` + 24 hex characters. Used `std::random_device` and `std::mt19937` for generation.

6. **Porting from Python to C++**: The parsers were ported from vLLM's Python implementation (Apache-2.0). Key differences:
   - Python's `re.DOTALL` → C++ `[\s\S]` pattern
   - Python's `json.loads()` → C++ `nlohmann::json::parse()`
   - Python's `uuid.uuid4().hex[:24]` → custom hex generation with `<random>`

---
