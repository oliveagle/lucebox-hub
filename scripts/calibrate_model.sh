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

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_PATH" ]; then
    echo "Usage: $0 <model_name_or_path> <output_stats_path> [options]"
    echo ""
    echo "Arguments:"
    echo "  model_name_or_path   - HuggingFace model name or local path"
    echo "  output_stats_path    - Output .pt file path for stats"
    echo ""
    echo "Options (passed through to calibrate.py):"
    echo "  --max-length INT     - Maximum token length (default: 32768)"
    echo "  --device DEVICE      - Device to run on (default: cuda)"
    echo "  --attn-implementation IMPL - Attention impl (default: flash_attention_2)"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen3-8B stats/qwen3_8b_stats.pt --max-length 32768"
    exit 1
fi

shift 2
ADDITIONAL_ARGS=("$@")

# Run calibration
python "$TRIATTENTION_ROOT/scripts/calibrate.py" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH" \
    "${ADDITIONAL_ARGS[@]}"
