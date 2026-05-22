#!/bin/bash
# Long context TriAttention performance test
# This script tests TriAttention compression on 8192+ token contexts

set -e

# Project root
cd /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash

# Paths
BUILD_DIR="./build"
TARGET_MODEL="../models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL="../models/draft/dflash-draft-3.6-q8_0.gguf"
LONG_PROMPT="/tmp/long_prompt_8192.bin"
OUTPUT_BASE="/tmp/long_context_test"

# Test configurations
CONTEXT_LENGTHS=(8192 16384)  # Test at 8K and 16K contexts
N_GEN=512  # Generate 512 tokens
KV_BUDGET=4096  # KV budget for TriAttention (50% of 8192)

echo "======================================"
echo "TriAttention Long Context Performance Test"
echo "======================================"
echo "Context lengths: ${CONTEXT_LENGTHS[@]}"
echo "Tokens to generate: $N_GEN"
echo "KV budget: $KV_BUDGET"
echo ""

# Generate long prompt
echo "Generating long prompt..."
python3 scripts/generate_long_prompt.py "$LONG_PROMPT" 8192

# Check prompt size
PROMPT_TOKENS=$(stat -c%s "$LONG_PROMPT" 2>/dev/null || stat -f%z "$LONG_PROMPT" 2>/dev/null)
PROMPT_TOKENS=$((PROMPT_TOKENS / 4))
echo "Prompt size: $PROMPT_TOKENS tokens"
echo ""

# Function to run DFlash test
run_dflash_test() {
    local label="$1"
    local tri_enabled="$2"
    local kv_budget="$3"
    local output_file="$4"

    echo "======================================"
    echo "Test: $label"
    echo "======================================"

    if [ "$tri_enabled" = "1" ]; then
        export TRIATTN_ENABLED=1
        export TRIATTN_KV_BUDGET=$kv_budget
        export TRIATTN_DIVIDE_LENGTH=128
        export TRIATTN_WINDOW_SIZE=128
        export TRIATTN_MIN_KEEP_RATIO=0.5
        echo "TriAttention: ENABLED (kv_budget=$kv_budget)"
    else
        unset TRIATTN_ENABLED
        echo "TriAttention: DISABLED (baseline)"
    fi

    cd "$BUILD_DIR"
    ./test_dflash \
        --max-ctx=16384 \
        "$TARGET_MODEL" \
        "$DRAFT_MODEL" \
        "$LONG_PROMPT" \
        "$N_GEN" \
        "$output_file" \
        2>&1 | tee "$output_file.log"
    cd - > /dev/null

    echo ""
}

# Function to run baseline (no DFlash) test
run_baseline_test() {
    local label="$1"
    local tri_enabled="$2"
    local kv_budget="$3"
    local output_file="$4"

    echo "======================================"
    echo "Test: $label"
    echo "======================================"

    if [ "$tri_enabled" = "1" ]; then
        export TRIATTN_ENABLED=1
        export TRIATTN_KV_BUDGET=$kv_budget
        export TRIATTN_DIVIDE_LENGTH=128
        export TRIATTN_WINDOW_SIZE=128
        export TRIATTN_MIN_KEEP_RATIO=0.5
        echo "TriAttention: ENABLED (kv_budget=$kv_budget)"
    else
        unset TRIATTN_ENABLED
        echo "TriAttention: DISABLED (baseline)"
    fi

    cd "$BUILD_DIR"
    ./test_generate \
        "$TARGET_MODEL" \
        "$LONG_PROMPT" \
        "$N_GEN" \
        "$output_file" \
        2>&1 | tee "$output_file.log"
    cd - > /dev/null

    echo ""
}

# Run tests for each context length
for ctx_len in "${CONTEXT_LENGTHS[@]}"; do
    echo "======================================"
    echo "Context Length: $ctx_len tokens"
    echo "======================================"

    # Regenerate prompt for this context length
    python3 scripts/generate_long_prompt.py "/tmp/long_prompt_${ctx_len}.bin" "$ctx_len"

    # Test 1: Baseline (no DFlash, no TriAttention)
    run_baseline_test \
        "Baseline (no DFlash, no TriAttention) - ${ctx_len} ctx" \
        "0" \
        "" \
        "${OUTPUT_BASE}_baseline_${ctx_len}.bin"

    # Test 2: DFlash only (no TriAttention)
    run_dflash_test \
        "DFlash only (no TriAttention) - ${ctx_len} ctx" \
        "0" \
        "" \
        "${OUTPUT_BASE}_dflash_${ctx_len}.bin"

    # Test 3: DFlash + TriAttention (kv_budget=4096 for 8192 ctx, 8192 for 16384 ctx)
    local budget=$((ctx_len / 2))
    run_dflash_test \
        "DFlash + TriAttention - ${ctx_len} ctx" \
        "1" \
        "$budget" \
        "${OUTPUT_BASE}_dflash_triattn_${ctx_len}.bin"

    echo ""
done

echo "======================================"
echo "All tests completed!"
echo "======================================"
echo ""
echo "Results are in ${OUTPUT_BASE}_*.bin.log"
echo ""
echo "To compare performance:"
echo "  grep 'tok/s' ${OUTPUT_BASE}_*.log"
echo "  grep 'TriAttention' ${OUTPUT_BASE}_*.log"
