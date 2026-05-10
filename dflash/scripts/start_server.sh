#!/bin/bash
# PFlash + DFlash OpenAI API Server 启动脚本
# 提供 OpenAI 兼容的 API 端点
#
# 现在支持两种模式：
# 1. Python HTTP Server + C++ Daemon (原有模式)
# 2. C++ HTTP Server (单一进程，推荐)

set -e

# 工作目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# 模型路径
TARGET_MODEL="./models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL="./models/draft/model.safetensors"
PFLASH_DRAFTER="./models/Qwen3-0.6B-BF16.gguf"

# 检查模型文件
check_model() {
    if [ ! -f "$1" ]; then
        echo "❌ 模型文件不存在: $1"
        exit 1
    fi
    local size=$(stat -c%s "$1" 2>/dev/null || stat -f%z "$1")
    local size_gb=$(echo "scale=2; $size / 1024 / 1024 / 1024" | bc)
    echo "✅ 模型文件: $1 (${size_gb} GB)"
}

echo "========================================"
echo "  PFlash + DFlash API Server 启动脚本"
echo "========================================"
echo ""

check_model "$TARGET_MODEL"
check_model "$DRAFT_MODEL"
check_model "$PFLASH_DRAFTER"

echo ""
echo "请选择服务器模式:"
echo "  1. Python HTTP Server + C++ Daemon (原有模式)"
echo "  2. C++ HTTP Server (单一进程，推荐，选项3功能)"
echo ""
read -p "请选择 (1-2): " mode_choice

if [ "$mode_choice" = "2" ]; then
    # ─────────────────────────────────────────────────────────────────
    # C++ HTTP Server 模式
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "C++ HTTP Server 模式"
    echo "===================="
    echo ""
    echo "配置选项:"
    echo "  1. 仅 DFlash (禁用 PFlash 压缩)"
    echo "  2. DFlash + PFlash (自动模式，阈值 8K tokens)"
    echo "  3. DFlash + PFlash (始终压缩) - 推荐"
    echo ""
    read -p "请选择启动模式 (1-3): " choice

    case $choice in
        1) PREFILL_MODE="off" ;;
        2) PREFILL_MODE="auto" ;;
        3) PREFILL_MODE="always" ;;
        *) echo "无效选择"; exit 1 ;;
    esac

    echo ""
    echo "启动 C++ HTTP 服务器..."
    echo "  - 目标模型: $TARGET_MODEL"
    echo "  - 起草模型: $DRAFT_MODEL"
    echo "  - PFlash 压缩: $PREFILL_MODE"
    if [ "$PREFILL_MODE" != "off" ]; then
        echo "  - PFlash 起草模型: $PFLASH_DRAFTER"
    fi
    echo "  - 端口: 8000"
    echo ""

    # 检查 dflash_server 是否已编译
    if [ ! -f "./build/dflash_server" ]; then
        echo "⚠️  C++ HTTP 服务器未编译"
        echo ""
        echo "请先运行:"
        echo "  cmake -B build -DDFLASH27B_HTTP_SERVER=ON"
        echo "  cmake --build build --target dflash_server"
        echo ""
        exit 1
    fi

    # 设置环境变量
    export DFLASH_FP_USE_BSA=1
    export DFLASH_FP_ALPHA=0.85
    export DFLASH27B_KV_TQ3=1
    export DFLASH27B_FA_WINDOW=2048

    # 构建命令
    CMD="./build/dflash_server \
        --target \"$TARGET_MODEL\" \
        --draft \"$DRAFT_MODEL\" \
        --host 0.0.0.0 \
        --port 8000 \
        --max-ctx 131072"

    if [ "$PREFILL_MODE" != "off" ]; then
        CMD="$CMD --prefill-$PREFILL_MODE"
        CMD="$CMD --prefill-drafter \"$PFLASH_DRAFTER\""
    fi

    echo "执行命令:"
    echo "$CMD"
    echo ""

    # 启动服务器
    eval exec $CMD

else
    # ─────────────────────────────────────────────────────────────────
    # Python HTTP Server + C++ Daemon 模式 (原有)
    # ─────────────────────────────────────────────────────────────────
    echo ""
    echo "Python HTTP Server + C++ Daemon 模式"
    echo "===================================="
    echo ""
    echo "配置选项:"
    echo "  1. 仅 DFlash (禁用 PFlash 压缩)"
    echo "  2. DFlash + PFlash (自动模式，阈值 8K tokens)"
    echo "  3. DFlash + PFlash (始终压缩)"
    echo "  4. 仅 AR 基线 (无推测解码)"
    echo ""
    read -p "请选择启动模式 (1-4): " choice

    case $choice in
        1) PREFILL_MODE="off" ;;
        2) PREFILL_MODE="auto" ;;
        3) PREFILL_MODE="always" ;;
        4) PREFILL_MODE="off"
           AR_ONLY="--ar-only"
           ;;
        *) echo "无效选择"; exit 1 ;;
    esac

    echo ""
    echo "启动 Python HTTP 服务器..."
    echo "  - 目标模型: $TARGET_MODEL"
    echo "  - 起草模型: $DRAFT_MODEL"
    echo "  - PFlash 压缩: $PREFILL_MODE"
    if [ "$PREFILL_MODE" != "off" ]; then
        echo "  - PFlash 起草模型: $PFLASH_DRAFTER"
    fi
    echo "  - 端口: 8000"
    echo ""

    export DFLASH_FP_USE_BSA=1
    export DFLASH_FP_ALPHA=0.85
    export DFLASH27B_KV_TQ3=1
    export DFLASH27B_FA_WINDOW=2048

    exec python3 scripts/server_tools.py \
        --target "$TARGET_MODEL" \
        --draft "$DRAFT_MODEL" \
        --port 8000 \
        --prefill-compression "$PREFILL_MODE" \
        --prefill-threshold 8192 \
        --prefill-keep-ratio 0.1 \
        --prefill-drafter "$PFLASH_DRAFTER" \
        $AR_ONLY
fi
