#!/bin/bash
# PFlash prefill 速度测试 - 使用 test_dflash daemon 的 compress 功能
# PFlash 通过 drafter scoring + compression 实现 ~10x TTFT 加速

set -e
cd /home/oliveagle/repos/github.com/others/lucebox-hub/dflash

# 清理 GPU
echo "清理 GPU 内存..."
pkill -9 test_dflash 2>/dev/null
pkill -9 python 2>/dev/null
sleep 3

# 确保 prompt 文件存在
if [ ! -f /tmp/prompt_64k.bin ]; then
    python3 -c "
import struct
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B', trust_remote_code=True)
text = '''深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。注意力机制允许模型动态关注输入的不同部分，Transformer 架构通过自注意力层实现了这一机制。'''
ids = tok.encode(text, add_special_tokens=False)
target = 65536
repeat = target // len(ids) + 1
ids_64k = (ids * repeat)[:target]
with open('/tmp/prompt_64k.bin', 'wb') as f:
    for t in ids_64k:
        f.write(struct.pack('<i', int(t)))
print(f'Created 64K: {len(ids_64k)} tokens')

target = 131072
repeat = target // len(ids) + 1
ids_128k = (ids * repeat)[:target]
with open('/tmp/prompt_128k.bin', 'wb') as f:
    for t in ids_128k:
        f.write(struct.pack('<i', int(t)))
print(f'Created 128K: {len(ids_128k)} tokens')
"
fi

echo "========================================"
echo "PFlash Prefill 速度测试"
echo "========================================"
echo ""

# 测试 64K PFlash
echo "===== 测试 64K PFlash ====="
echo ""
{
    # 1. Park target + draft 释放内存
    echo "park"
    sleep 2
    # 2. Compress with PFlash (keep_ratio=0.10)
    echo "compress /tmp/prompt_64k.bin 100 models/Qwen3-0.6B-BF16.gguf"
    sleep 15
    # 3. 清理
    echo "exit"
} | DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
    ./build/test_dflash \
    models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/model.safetensors \
    --daemon 2>&1 | tee /tmp/pflash_64k.log

echo ""
echo "64K PFlash 结果:"
grep -E "\[compress\]|\[drafter\] forward" /tmp/pflash_64k.log | tail -5

echo ""
echo "清理 GPU..."
pkill -9 test_dflash 2>/dev/null
sleep 5

# 测试 128K PFlash
echo ""
echo "===== 测试 128K PFlash ====="
echo ""
{
    echo "park"
    sleep 2
    echo "compress /tmp/prompt_128k.bin 100 models/Qwen3-0.6B-BF16.gguf"
    sleep 20
    echo "exit"
} | DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
    ./build/test_dflash \
    models/Qwen3.6-27B-Q4_K_M.gguf \
    models/draft/model.safetensors \
    --daemon 2>&1 | tee /tmp/pflash_128k.log

echo ""
echo "128K PFlash 结果:"
grep -E "\[compress\]|\[drafter\] forward" /tmp/pflash_128k.log | tail -5

echo ""
echo "========================================"
echo "PFlash 测试完成！"
echo "========================================"
echo ""
echo "日志文件:"
echo "  64K:  /tmp/pflash_64k.log"
echo "  128K: /tmp/pflash_128k.log"
echo ""

# 解析速度
echo "===== 速度解析 ====="
echo ""
echo "64K context:"
grep "\[compress\]" /tmp/pflash_64k.log | grep -v "\[compress\] target\|\[compress\] draft\|\[compress\] bad\|\[compress\] empty" | awk '{print "  Original: "$1" -> Compressed: "$3" tokens  Time: "$5"s  Speed: "$3/$5" tok/s"}'
echo ""
echo "128K context:"
grep "\[compress\]" /tmp/pflash_128k.log | grep -v "\[compress\] target\|\[compress\] draft\|\[compress\] bad\|\[compress\] empty" | awk '{print "  Original: "$1" -> Compressed: "$3" tokens  Time: "$5"s  Speed: "$3/$5" tok/s"}'