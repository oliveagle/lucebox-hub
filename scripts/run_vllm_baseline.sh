#!/bin/bash
# Run vLLM baseline benchmark WITHOUT TriAttention KV compression
#
# This script launches a vLLM server without TriAttention and runs
# a performance benchmark to measure prefill speed, decode speed,
# and first token latency.
#
# Usage:
#   ./scripts/run_vllm_baseline.sh <model_path> [additional_vllm_args...]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
MODEL_PATH="${1:-}"
if [ -z "$MODEL_PATH" ]; then
    echo "Usage: $0 <model_path> [additional_vllm_args...]"
    echo ""
    echo "Example:"
    echo "  $0 Qwen/Qwen3.6-27B-AWQ --max-model-len 4096"
    echo "  $0 models/Qwen3.6-27B-AWQ --gpu-memory-utilization 0.85"
    exit 1
fi

shift
ADDITIONAL_ARGS=("$@")

# Disable TriAttention
export ENABLE_TRIATTENTION="false"
export TRIATTN_RUNTIME_KV_BUDGET=""

echo "======================================"
echo "vLLM Baseline Benchmark"
echo "======================================"
echo "Model: $MODEL_PATH"
echo "TriAttention: DISABLED (baseline)"
echo "Additional args: ${ADDITIONAL_ARGS[*]}"
echo "======================================"

# Launch vLLM server in background
# Note: Not using --enforce-eager for baseline to allow CUDA graphs
echo "Starting vLLM server..."
vllm serve "$MODEL_PATH" \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port 8000 \
    "${ADDITIONAL_ARGS[@]}" \
    > /tmp/vllm_baseline.log 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "Timeout waiting for server to start"
        cat /tmp/vllm_baseline.log
        kill $VLLM_PID 2>/dev/null || true
        exit 1
    fi
    sleep 2
done

# Run benchmark
echo ""
echo "======================================"
echo "Running benchmark..."
echo "======================================"

source /mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/.venv/bin/activate
python3 "$SCRIPT_DIR/benchmark_vllm.py" \
    --model "$MODEL_PATH" \
    --base-url http://localhost:8000

# Cleanup
echo ""
echo "Shutting down server..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo "Baseline benchmark complete!"
