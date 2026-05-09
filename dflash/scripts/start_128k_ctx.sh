#!/bin/bash
# 启动 128K context 测试的服务器

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  PFlash + DFlash - 128K Context 测试"
echo "========================================"
echo ""

# 检查模型文件
for model in "models/Qwen3.6-27B-Q4_K_M.gguf" "models/draft/model.safetensors" "models/Qwen3-0.6B-BF16.gguf"; do
    if [ ! -f "$model" ]; then
        echo "❌ 模型文件不存在: $model"
        exit 1
    fi
done

echo "✅ 所有模型文件已找到"
echo ""

# 设置环境变量
export DFLASH_FP_USE_BSA=1
export DFLASH_FP_ALPHA=0.85
export DFLASH27B_KV_TQ3=1
export DFLASH27B_FA_WINDOW=2048

echo "配置:"
echo "  Max Context: 131072 (128K)"
echo "  KV TQ3_0: Yes"
echo "  FA Window: 2048"
echo "  Budget: 22"
echo ""

# 启动服务器
echo "启动服务器..."
exec python3 scripts/server_tools.py \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft \
    --bin build/test_dflash \
    --budget 22 \
    --max-ctx 131072 \
    --port 8000 \
    --prefill-compression auto \
    --prefill-threshold 4096 \
    --prefill-keep-ratio 0.1 \
    --prefill-drafter models/Qwen3-0.6B-BF16.gguf \
    --tokenizer Qwen/Qwen3.6-27B
