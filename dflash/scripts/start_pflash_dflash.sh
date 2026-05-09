#!/bin/bash
# PFlash + DFlash API 服务启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  PFlash + DFlash API Server"
echo "========================================"
echo ""
echo "工作目录: $(pwd)"
echo ""

# 检查模型文件
echo "检查模型文件..."
for model in "models/Qwen3.6-27B-Q4_K_M.gguf" "models/draft/model.safetensors" "models/Qwen3-0.6B-BF16.gguf"; do
    if [ -f "$model" ]; then
        echo "  ✅ $model"
    else
        echo "  ❌ $model 不存在"
        exit 1
    fi
done
echo ""

# 设置环境变量
export DFLASH_FP_USE_BSA=1
export DFLASH_FP_ALPHA=0.85
export DFLASH27B_KV_TQ3=1
export DFLASH27B_FA_WINDOW=2048

echo "启动配置:"
echo "  - PFlash 压缩: auto (阈值 4K tokens)"
echo "  - PFlash 保留率: 10%"
echo "  - DDTree budget: 22"
echo "  - Max context: 131072 (128K)"
echo "  - KV cache: TQ3_0"
echo "  - FA window: 2048"
echo ""

# 启动服务器
exec python3 scripts/server_tools.py \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft \
    --bin build/test_dflash \
    --budget 22 \
    --max-ctx 16384 \
    --port 8000 \
    --prefill-compression auto \
    --prefill-threshold 4096 \
    --prefill-keep-ratio 0.1 \
    --prefill-drafter models/Qwen3-0.6B-BF16.gguf \
    --tokenizer Qwen/Qwen3.6-27B
