#!/bin/bash
# 测试最大 context 大小的服务器启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "  PFlash + DFlash - 最大 Context 测试"
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

# 选择 context 大小
echo "选择要测试的 context 大小:"
echo "  1. 32K  (默认，适合大多数场景)"
echo "  2. 64K  (长上下文)"
echo "  3. 128K (超长上下文)"
echo "  4. 256K (最大上下文，需要 TQ3_0 KV)"
echo ""
read -p "请选择 (1-4): " choice

case $choice in
    1)
        MAX_CTX=32768
        KV_TQ3=0
        ;;
    2)
        MAX_CTX=65536
        KV_TQ3=1
        ;;
    3)
        MAX_CTX=131072
        KV_TQ3=1
        ;;
    4)
        MAX_CTX=262144
        KV_TQ3=1
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac

echo ""
echo "配置:"
echo "  Max Context: $MAX_CTX"
echo "  KV TQ3_0: $KV_TQ3"
echo ""

# 设置环境变量
export DFLASH_FP_USE_BSA=1
export DFLASH_FP_ALPHA=0.85
if [ "$KV_TQ3" = "1" ]; then
    export DFLASH27B_KV_TQ3=1
else
    unset DFLASH27B_KV_TQ3
fi
export DFLASH27B_FA_WINDOW=2048

# 启动服务器
echo "启动服务器..."
exec python3 scripts/server_tools.py \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft \
    --bin build/test_dflash \
    --budget 22 \
    --max-ctx $MAX_CTX \
    --port 8000 \
    --prefill-compression auto \
    --prefill-threshold 4096 \
    --prefill-keep-ratio 0.1 \
    --prefill-drafter models/Qwen3-0.6B-BF16.gguf \
    --tokenizer Qwen/Qwen3.6-27B
