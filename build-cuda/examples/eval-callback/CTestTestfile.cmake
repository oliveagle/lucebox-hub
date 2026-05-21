# CMake generated Testfile for 
# Source directory: /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/examples/eval-callback
# Build directory: /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/build-cuda/examples/eval-callback
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test-eval-callback-download-model]=] "/home/oliveagle/.local/lib/python3.12/site-packages/cmake/data/bin/cmake" "-DDEST=/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/build-cuda/tinyllamas/stories15M-q4_0.gguf" "-DNAME=tinyllamas/stories15M-q4_0.gguf" "-DHASH=SHA256=66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739" "-P" "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/cmake/download-models.cmake")
set_tests_properties([=[test-eval-callback-download-model]=] PROPERTIES  FIXTURES_SETUP "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/examples/eval-callback/CMakeLists.txt;17;add_test;/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/examples/eval-callback/CMakeLists.txt;0;")
add_test([=[test-eval-callback]=] "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/build-cuda/bin/llama-eval-callback" "-m" "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/build-cuda/tinyllamas/stories15M-q4_0.gguf" "--prompt" "hello" "--seed" "42" "-ngl" "0")
set_tests_properties([=[test-eval-callback]=] PROPERTIES  FIXTURES_REQUIRED "test-eval-callback-download-model" _BACKTRACE_TRIPLES "/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/examples/eval-callback/CMakeLists.txt;24;add_test;/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/deps/llama.cpp/examples/eval-callback/CMakeLists.txt;0;")
