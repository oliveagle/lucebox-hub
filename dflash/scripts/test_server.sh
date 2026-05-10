#!/bin/bash
# 简单测试脚本，验证 dflash_server 能正常启动和响应

set -e

echo "========================================"
echo "  DFlash C++ HTTP 服务器测试"
echo "========================================"
echo ""

# 检查二进制文件
if [ ! -f "./build/dflash_server" ]; then
    echo "❌ dflash_server 未编译"
    echo "请先运行: cmake --build build --target dflash_server"
    exit 1
fi

echo "✅ dflash_server 已编译"
echo ""

# 检查模型文件（测试用虚拟路径）
TARGET_MODEL="./models/Qwen3.6-27B-Q4_K_M.gguf"
DRAFT_MODEL="./models/draft/model.safetensors"

echo "检查模型文件..."
if [ ! -f "$TARGET_MODEL" ]; then
    echo "⚠️  目标模型不存在: $TARGET_MODEL"
    echo "这是预期的 - 我们只测试服务器能否启动"
fi

if [ ! -f "$DRAFT_MODEL" ]; then
    echo "⚠️  起草模型不存在: $DRAFT_MODEL"
    echo "这是预期的 - 我们只测试服务器能否启动"
fi

echo ""
echo "测试 1: 帮助信息"
echo "==============="
./build/dflash_server --help
echo ""

echo "测试 2: 缺少必需参数时的错误处理"
echo "===================================="
if ./build/dflash_server 2>&1 | grep -q "Error: --target is required"; then
    echo "✅ 正确返回错误信息"
    EXIT_CODE=0
else
    echo "❌ 未返回预期的错误信息"
    EXIT_CODE=1
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 测试通过"
else
    echo "❌ 测试失败"
    exit 1
fi

echo ""
echo "========================================"
echo "  所有测试通过！"
echo "========================================"
echo ""
echo "下一步:"
echo "1. 下载模型文件到 models/ 目录"
echo "2. 运行: ./build/dflash_server --target models/Qwen3.6-27B-Q4_K_M.gguf --draft models/draft/model.safetensors"
echo "3. 测试 API: curl http://localhost:8000/v1/models"
