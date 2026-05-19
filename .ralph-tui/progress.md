# Ralph Progress Log

This file tracks progress across iterations. Agents update this file
after each iteration and it's included in prompts for context.

## Codebase Patterns (Study These First)

*Add reusable patterns discovered during development here.*

---

## [2026-05-19] - lucebox-hub-gfx1151-1oe.2
- **What was implemented**:
  - Added `common_triattention.h` and `common_triattention.cpp` to llama.cpp common library
  - Created `common_triattention_config` class that reads TriAttention settings from environment variables:
    - `TRIATTN_STATS_PATH` — path to .bin stats file
    - `TRIATTN_KV_BUDGET` — max tokens to retain (default 2048)
    - `TRIATTN_DIVIDE_LENGTH` — compression interval (default 128)
    - `TRIATTN_WINDOW_SIZE` — recent tokens preserved (default 128)
    - `TRIATTN_ENABLED` — master switch (default: auto if stats path is set)
  - Updated `common/CMakeLists.txt` to conditionally build and link with `triattention` library
  - Added TriAttention startup logging in `tools/server/server.cpp`
  - Set `GGML_TRIATTENTION` as PUBLIC compile definition in common library
- **Files changed**:
  - `dflash/deps/llama.cpp/common/common_triattention.h` (new)
  - `dflash/deps/llama.cpp/common/common_triattention.cpp` (new)
  - `dflash/deps/llama.cpp/common/CMakeLists.txt` (modified)
  - `dflash/deps/llama.cpp/tools/server/server.cpp` (modified)
- **Learnings**:
  - CMake `target_compile_definitions(PRIVATE ...)` doesn't propagate to dependent targets; use `PUBLIC` for definitions needed by consumers
  - The TriAttention C library (`libtriattention.a`) is conditionally built based on `GGML_TRIATTENTION` or `LLAMA_TRIATTENTION` flags
  - llama.cpp fork uses GGML backend; DFlash uses its own custom inference engine
  - Environment variable reading should use `std::getenv()` with fallback defaults for robust configuration

---

