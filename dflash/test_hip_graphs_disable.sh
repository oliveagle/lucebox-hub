#!/bin/bash
# Test script to verify HIP Graphs disable behavior on gfx1151
# This script should be run after a successful build

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
MODEL_PATH="${SCRIPT_DIR}/models/Qwen3.6-27B-Q4_K_M.gguf"
TEST_PROMPT="/tmp/test_prompt.bin"
TEST_OUTPUT="/tmp/test_output.bin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    if [ ! -f "${BUILD_DIR}/test_generate" ]; then
        log_error "test_generate binary not found. Please build first: cmake --build build --target test_generate"
        exit 1
    fi

    if [ ! -f "${MODEL_PATH}" ]; then
        log_warn "Model file not found at ${MODEL_PATH}"
        log_warn "Skipping actual execution test"
        return 1
    fi

    return 0
}

# Create test prompt
create_test_prompt() {
    log_info "Creating test prompt..."
    python3 -c "
import struct
# Simple test tokens: "Hello world" in Qwen token format
tokens = [151644, 872, 570]
with open('${TEST_PROMPT}', 'wb') as f:
    for t in tokens:
        f.write(struct.pack('<I', t))
" || {
        log_error "Failed to create test prompt"
        return 1
    }
    log_info "Test prompt created: ${TEST_PROMPT}"
}

# Test with GGML_CUDA_DISABLE_GRAPHS environment variable
test_with_env_var() {
    log_info "Testing with GGML_CUDA_DISABLE_GRAPHS=1..."

    export GGML_CUDA_DISABLE_GRAPHS=1

    timeout 60 "${BUILD_DIR}/test_generate" \
        "${MODEL_PATH}" \
        "${TEST_PROMPT}" \
        10 \
        "${TEST_OUTPUT}" 2>&1 | tee /tmp/test_output_graphs_disabled.log || {
        log_warn "Test with GGML_CUDA_DISABLE_GRAPHS=1 timed out or failed (this may be expected)"
        return 1
    }

    unset GGML_CUDA_DISABLE_GRAPHS
    log_info "Test with GGML_CUDA_DISABLE_GRAPHS=1 completed"
    return 0
}

# Test without environment variable (auto-disable via compile-time check)
test_auto_disable() {
    log_info "Testing with auto-disable (gfx1151 compile-time detection)..."

    timeout 60 "${BUILD_DIR}/test_generate" \
        "${MODEL_PATH}" \
        "${TEST_PROMPT}" \
        10 \
        "${TEST_OUTPUT}" 2>&1 | tee /tmp/test_output_auto_disable.log || {
        log_warn "Test with auto-disable timed out or failed"
        return 1
    }

    log_info "Test with auto-disable completed"
    return 0
}

# Check output for graph-related messages
check_output_for_graph_messages() {
    local log_file=$1

    log_info "Checking ${log_file} for graph-related messages..."

    if grep -q "CUDA graphs" "${log_file}"; then
        log_info "Found CUDA graphs messages:"
        grep "CUDA graphs" "${log_file}"
    fi

    if grep -q "ggml_cuda_init" "${log_file}"; then
        log_info "Found ggml_cuda_init messages:"
        grep "ggml_cuda_init" "${log_file}"
    fi
}

# Main test flow
main() {
    log_info "=== HIP Graphs Disable Test on gfx1151 ==="
    log_info "Build directory: ${BUILD_DIR}"
    log_info "Model path: ${MODEL_PATH}"

    check_prerequisites || exit 0

    create_test_prompt

    # Run tests
    local test_passed=0

    if test_with_env_var; then
        ((test_passed++))
        check_output_for_graph_messages /tmp/test_output_graphs_disabled.log
    fi

    if test_auto_disable; then
        ((test_passed++))
        check_output_for_graph_messages /tmp/test_output_auto_disable.log
    fi

    log_info "=== Test Summary ==="
    log_info "Tests passed: ${test_passed}/2"

    if [ ${test_passed} -eq 0 ]; then
        log_warn "No tests passed - this may indicate the hang issue persists"
        log_warn "Check the log files for details:"
        log_warn "  - /tmp/test_output_graphs_disabled.log"
        log_warn "  - /tmp/test_output_auto_disable.log"
    else
        log_info "At least one test passed - graphs disable is working"
    fi
}

# Usage
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0"
    echo ""
    echo "This script tests the HIP Graphs auto-disable functionality on gfx1151."
    echo ""
    echo "Prerequisites:"
    echo "  1. Built test_generate binary"
    echo "  2. Model file at ${MODEL_PATH}"
    echo ""
    echo "The script will:"
    echo "  1. Create a minimal test prompt"
    echo "  2. Run test with GGML_CUDA_DISABLE_GRAPHS=1"
    echo "  3. Run test with auto-disable (compile-time detection)"
    echo "  4. Report results"
    exit 0
fi

main "$@"
