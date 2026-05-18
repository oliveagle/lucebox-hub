#!/bin/bash
# Calibrate TriAttention frequency statistics for a model
#
# This script runs a forward pass on a model to compute per-head frequency
# statistics needed for TriAttention KV cache compression.
#
# Usage:
#   ./scripts/calibrate_model.sh <model_name_or_path> <output_stats_path> [options]
#
# Example:
#   ./scripts/calibrate_model.sh Qwen/Qwen3-8B triattention_stats.pt --max-length 32768

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

# Check if triattention is installed
if ! python -c "import triattention" 2>/dev/null; then
    echo "Installing TriAttention from submodule..."
    pip install -e "$TRIATTENTION_ROOT"
fi

# Parse arguments
MODEL_PATH="${1:-}"
OUTPUT_PATH="${2:-}"
shift 2

# Parse options for calibrate.py
INPUT_PATH=""
MAX_LENGTH=""
DEVICE=""
ATTN_IMPL=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_PATH="$2"
            shift 2
            ;;
        --max-length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$1"
            DEVICE_VALUE="$2"
            shift 2
            ;;
        --attn-implementation)
            ATTN_IMPL="$1"
            ATTN_IMPL_VALUE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$INPUT_PATH" ]; then
    echo "Usage: $0 <model_name_or_path> <output_stats_path> --input <input_file> [options]"
    echo ""
    echo "Arguments:"
    echo "  model_name_or_path   - HuggingFace model name or local path"
    echo "  output_stats_path    - Output .pt file path for stats"
    echo ""
    echo "Required options:"
    echo "  --input INPUT        - Plain text file for calibration"
    echo ""
    echo "Optional options (passed through to calibrate.py):"
    echo "  --max-length INT     - Maximum token length (default: 32768)"
    echo "  --device DEVICE      - Device to run on (default: cuda)"
    echo "  --attn-implementation IMPL - Attention impl (default: flash_attention_2)"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen3-8B stats/qwen3_8b_stats.pt --input calibration.txt --max-length 32768"
    exit 1
fi

# Build additional args
ADDITIONAL_ARGS=()
[ -n "$MAX_LENGTH" ] && ADDITIONAL_ARGS+=("--max-length" "$MAX_LENGTH")
[ -n "$DEVICE" ] && ADDITIONAL_ARGS+=("$DEVICE" "$DEVICE_VALUE")
[ -n "$ATTN_IMPL" ] && ADDITIONAL_ARGS+=("$ATTN_IMPL" "$ATTN_IMPL_VALUE")

# Run calibration
python "$TRIATTENTION_ROOT/scripts/calibrate.py" \
    --model "$MODEL_PATH" \
    --input "$INPUT_PATH" \
    --output "$OUTPUT_PATH" \
    "${ADDITIONAL_ARGS[@]}"
