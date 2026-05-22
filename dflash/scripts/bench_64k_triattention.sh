#!/usr/bin/env bash
set -e

echo "=========================================="
echo " 64K Long Context TriAttention Benchmark"
echo " GPU: AMD gfx1151 (Strix Halo)"
echo " Date: $(date)"
echo "=========================================="

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DFLASH_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_TRIATT="$DFLASH_DIR/build-triatt"
TARGET_MODEL="/mnt/eaget-4tb/modelscope_models/unsloth/Qwen3___6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL="/mnt/eaget-4tb/modelscope_models/z-lab/Qwen3___6-27B-DFlash/model.safetensors"
OUTPUT_DIR="/tmp/triattention_64k_bench"
PROMPT_64K="$OUTPUT_DIR/prompt_65536.bin"
mkdir -p "$OUTPUT_DIR"

# Generate 65536 token prompt
echo "[1/3] Generating 65536 token prompt..."
python3 "$SCRIPT_DIR/generate_long_prompt.py" "$PROMPT_64K" 65536

# Environment
export ROCM_PATH=~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH=~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_core/lib:~/venvs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:$LD_LIBRARY_PATH
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # gfx1151 override

NUM_GEN=256  # generate 256 tokens

echo ""
echo "[2/3] Running DFlash only (no TriAttention) at 64K context..."
echo "-----------------------------------------------------------"

# Build DFlash only binary (no TriAttention)
BUILD_NO_TRIATT="$DFLASH_DIR/build-dflash-only"
if [ ! -f "$BUILD_NO_TRIATT/test_dflash" ]; then
    echo "  Building DFlash only binary..."
    cmake -B "$BUILD_NO_TRIATT" -S "$DFLASH_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DDFLASH27B_GPU_BACKEND=hip \
        -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
        -DDFLASH27B_HIP_SM80_EQUIV=ON \
        -DDFLASH27B_TRIATTENTION=OFF 2>&1 | tail -5
    cmake --build "$BUILD_NO_TRIATT" --target test_dflash -j$(nproc) 2>&1 | tail -5
fi

DFLASH_ONLY_OUTPUT="$OUTPUT_DIR/dflash_only.log"
"$BUILD_NO_TRIATT/test_dflash" \
    "$TARGET_MODEL" \
    "$DRAFT_MODEL" \
    "$PROMPT_64K" \
    "$NUM_GEN" \
    "$OUTPUT_DIR/dflash_only_output.bin" \
    --max-ctx 65536 \
    --fa-window 2048 \
    --ddtree-budget 22 \
    --kv-tq3 \
    2>&1 | tee "$DFLASH_ONLY_OUTPUT"

echo ""
echo "[3/3] Running DFlash + TriAttention at 64K context..."
echo "-----------------------------------------------------------"

# Build TriAttention binary
if [ ! -f "$BUILD_TRIATT/test_dflash" ]; then
    echo "  Building DFlash+TriAttention binary..."
    cmake -B "$BUILD_TRIATT" -S "$DFLASH_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DDFLASH27B_GPU_BACKEND=hip \
        -DDFLASH27B_HIP_ARCHITECTURES=gfx1151 \
        -DDFLASH27B_HIP_SM80_EQUIV=ON \
        -DDFLASH27B_TRIATTENTION=ON 2>&1 | tail -5
    cmake --build "$BUILD_TRIATT" --target test_dflash -j$(nproc) 2>&1 | tail -5
fi

TRIATT_OUTPUT="$OUTPUT_DIR/dflash_triattention.log"
"$BUILD_TRIATT/test_dflash" \
    "$TARGET_MODEL" \
    "$DRAFT_MODEL" \
    "$PROMPT_64K" \
    "$NUM_GEN" \
    "$OUTPUT_DIR/triattention_output.bin" \
    --max-ctx 65536 \
    --fa-window 2048 \
    --ddtree-budget 22 \
    --kv-tq3 \
    --tri-kv-budget 4096 \
    --tri-min-keep 0.75 \
    2>&1 | tee "$TRIATT_OUTPUT"

echo ""
echo "=========================================="
echo "  Results Summary"
echo "=========================================="
echo ""

# Parse decode speed from logs
parse_decode_speed() {
    local log_file="$1"
    grep -i "decode\|tok/s\|tokens/s\|speed" "$log_file" | tail -5 || echo "  (no decode speed found)"
}

echo "--- DFlash only (64K context) ---"
parse_decode_speed "$DFLASH_ONLY_OUTPUT"
echo ""
echo "--- DFlash + TriAttention (64K context, kv_budget=4096) ---"
parse_decode_speed "$TRIATT_OUTPUT"

echo ""
echo "Full logs saved to:"
echo "  DFlash only: $DFLASH_ONLY_OUTPUT"
echo "  TriAttention: $TRIATT_OUTPUT"
echo ""
echo "Output files saved to: $OUTPUT_DIR"
echo "=========================================="
