#!/bin/bash
# Run vLLM with TriAttention KV compression
#
# This script launches a vLLM server with TriAttention KV cache compression
# enabled. The TriAttention vLLM plugin is auto-discovered when installed.
#
# Environment variables:
#   TRIATTN_RUNTIME_KV_BUDGET         - Maximum tokens retained per request (default: 2048)
#   TRIATTN_RUNTIME_DIVIDE_LENGTH     - Compression trigger interval (default: 128)
#   TRIATTN_RUNTIME_WINDOW_SIZE       - Recent tokens always preserved (default: 128)
#   TRIATTN_RUNTIME_SPARSE_STATS_PATH - Path to precomputed frequency statistics (.pt file)
#   ENABLE_TRIATTENTION               - Master switch (default: true)
#
# Usage:
#   ./scripts/run_vllm_with_triattention.sh <model_path> [additional_vllm_args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TRIATTENTION_ROOT="$PROJECT_ROOT/submodules/triattention"

# Check if TriAttention submodule is initialized
if [ ! -d "$TRIATTENTION_ROOT" ]; then
    echo "Error: TriAttention submodule not found at $TRIATTENTION_ROOT"
    echo "Run: git submodule update --init --recursive"
    exit 1
fi

# Default values
KV_BUDGET="${TRIATTN_RUNTIME_KV_BUDGET:-2048}"
DIVIDE_LENGTH="${TRIATTN_RUNTIME_DIVIDE_LENGTH:-128}"
WINDOW_SIZE="${TRIATTN_RUNTIME_WINDOW_SIZE:-128}"
STATS_PATH="${TRIATTN_RUNTIME_SPARSE_STATS_PATH:-}"
ENABLE_TRIATTENTION="${ENABLE_TRIATTENTION:-true}"

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "Error: vllm is not installed"
    echo "Install with: pip install vllm"
    exit 1
fi

# Check if TriAttention is installed
if ! python -c "import triattention" 2>/dev/null; then
    echo "TriAttention not installed. Installing from submodule..."
    pip install -e "$TRIATTENTION_ROOT"
    if [ -n "$STATS_PATH" ]; then
        pip install flash-attn --no-build-isolation
    fi
fi

# Validate STATS_PATH if provided
if [ -n "$STATS_PATH" ] && [ ! -f "$STATS_PATH" ]; then
    echo "Warning: Stats file not found at $STATS_PATH"
    echo "KV compression may not work correctly."
    echo "Generate stats with: ./scripts/calibrate_model.sh"
fi

# Set environment variables
export TRIATTN_RUNTIME_KV_BUDGET="$KV_BUDGET"
export TRIATTN_RUNTIME_DIVIDE_LENGTH="$DIVIDE_LENGTH"
export TRIATTN_RUNTIME_WINDOW_SIZE="$WINDOW_SIZE"
export TRIATTN_RUNTIME_SPARSE_STATS_PATH="$STATS_PATH"
export ENABLE_TRIATTENTION="$ENABLE_TRIATTENTION"

# Default vLLM arguments
MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [additional_vllm_args...]"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen3-8B --max-model-len 32768"
    echo ""
    echo "Environment variables:"
    echo "  TRIATTN_RUNTIME_KV_BUDGET=$KV_BUDGET"
    echo "  TRIATTN_RUNTIME_DIVIDE_LENGTH=$DIVIDE_LENGTH"
    echo "  TRIATTN_RUNTIME_WINDOW_SIZE=$WINDOW_SIZE"
    echo "  TRIATTN_RUNTIME_SPARSE_STATS_PATH=${STATS_PATH:-<not set>}"
    exit 1
fi

shift
ADDITIONAL_ARGS=("$@")

# Launch vLLM server
# Key settings for TriAttention:
#   --enforce-eager: required (no CUDA graph support yet)
#   --enable-prefix-caching false: incompatible with KV compression
echo "Starting vLLM with TriAttention..."
echo "  KV Budget: $KV_BUDGET"
echo "  Stats Path: ${STATS_PATH:-<none>}"
echo "  Model: $MODEL_PATH"

vllm serve "$MODEL_PATH" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --enforce-eager \
    --trust-remote-code \
    --enable-prefix-caching false \
    "${ADDITIONAL_ARGS[@]}"
