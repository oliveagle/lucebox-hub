#!/bin/bash
# 64K 和 128K Prefill 速度测试

cd /home/oliveagle/repos/github.com/others/lucebox-hub/dflash

echo "========================================"
echo "Prefill 速度测试"
echo "========================================"
echo ""

# 清理并检查 GPU
echo "1. 清理 GPU 内存..."
pkill -9 test_dflash 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3
echo ""

# 测试 64K (增加 padding)
echo "========================================"
echo "测试 64K context..."
echo "========================================"
DFLASH27B_KV_TQ3=1 ./build/test_dflash \
    models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/model.safetensors \
    /tmp/prompt_64k.bin \
    32 \
    /tmp/out_64k.bin \
    --max-ctx=65600 \
    --fast-rollback 2>&1 | tee /tmp/result_64k.log

# 清理后测试 128K
echo ""
echo "清理 GPU 后测试 128K..."
pkill -9 test_dflash 2>/dev/null
sleep 5
echo ""
echo "========================================"
echo "测试 128K context..."
echo "========================================"
DFLASH27B_KV_TQ3=1 ./build/test_dflash \
    models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/model.safetensors \
    /tmp/prompt_128k.bin \
    32 \
    /tmp/out_128k.bin \
    --max-ctx=131200 \
    --fast-rollback 2>&1 | tee /tmp/result_128k.log

echo ""
echo "========================================"
echo "测试完成！解析结果..."
echo "========================================"
echo ""

# 解析结果
echo "64K context 结果:"
echo "----------------"
grep "\[prefill\]" /tmp/result_64k.log

echo ""
echo "128K context 结果:"
echo "----------------"
grep "\[prefill\]" /tmp/result_128k.log

echo ""
echo "========================================"
echo "速度汇总"
echo "========================================"
echo -n "64K:  "
grep "\[prefill\]" /tmp/result_64k.log | awk '{print "Tokens: "$2"  Time: "$4"s  Speed: "$2/$4" tok/s"}'

echo -n "128K: "
grep "\[prefill\]" /tmp/result_128k.log | awk '{print "Tokens: "$2"  Time: "$4"s  Speed: "$2/$4" tok/s"}'