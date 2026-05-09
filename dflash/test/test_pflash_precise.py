#!/usr/bin/env python3
"""
PFlash + DFlash 256K Context 性能测试
精确测试：预热 -> 清理缓存 -> 正式测试
"""
import requests
import json
import time
import sys
from pathlib import Path
from transformers import AutoTokenizer

def create_long_prompt(target_tokens, tokenizer):
    """创建指定 token 数量的长文本"""
    base = (
        "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。"
        "深度学习模型通过多层神经网络结构，能够自动从数据中学习特征表示。"
        "卷积神经网络（CNN）主要用于处理图像数据，通过卷积层提取局部特征。"
        "循环神经网络（RNN）适合处理序列数据，如文本和时间序列。"
        "Transformer 架构引入了自注意力机制，能够并行处理序列数据。"
        "大语言模型（LLM）基于 Transformer 架构，在海量文本数据上训练。"
        "Qwen 是由阿里巴巴开发的大语言模型系列，性能优异。"
        "推测解码（Speculative Decoding）使用小型模型加速大型模型的推理。"
        "PFlash 是一种预填充压缩技术，能够显著减少长上下文的处理时间。"
        "DFlash 结合了块扩散推测解码和 DDTree 树状验证。"
    )

    base_tokens = len(tokenizer.encode(base, add_special_tokens=False))
    repeat = max(1, (target_tokens // base_tokens) - 1)
    text = " ".join([base] * repeat)

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=False)

    return text

def test_precise(url, prompt, tokenizer, label=""):
    """精确测试：预热 -> 清理 -> 测试"""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")

    original_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"  原始 tokens: {original_tokens}")

    # 步骤 1: 预热（短 prompt）
    print("\n  [1/3] 预热...")
    warmup_payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 32,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    resp = requests.post(f"{url}/v1/chat/completions", json=warmup_payload, stream=True, timeout=60)
    for line in resp.iter_lines():
        if line and b'data:' in line:
            break
    print("  ✅ 预热完成")

    # 步骤 2: 测试 Prefill（非流式）
    print("\n  [2/3] 测试 Prefill...")
    prefill_payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    prefill_start = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=prefill_payload, timeout=600)
    prefill_time = time.time() - prefill_start

    if resp.status_code != 200:
        print(f"  ❌ 错误: {resp.status_code} - {resp.text}")
        return None

    result = resp.json()
    usage = result.get('usage', {})
    compressed_tokens = usage.get('prompt_tokens', 0)
    compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0

    print(f"  压缩后 tokens: {compressed_tokens}")
    print(f"  压缩比: {compression_ratio:.1f}:1")
    print(f"  Prefill 时间: {prefill_time:.2f} s")
    print(f"  Prefill 速度: {original_tokens / prefill_time:.1f} tok/s")

    # 步骤 3: 测试 Decode（流式）
    print("\n  [3/3] 测试 Decode...")
    decode_payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    decode_start = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=decode_payload, stream=True, timeout=600)

    if resp.status_code != 200:
        print(f"  ❌ 错误: {resp.status_code} - {resp.text}")
        return None

    first_token_time = None
    token_count = 0
    content = ""

    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    if first_token_time is None and data.get('choices', [{}])[0].get('delta', {}).get('content'):
                        first_token_time = time.time()
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    if delta.get('content'):
                        content += delta['content']
                        token_count += 1
                except Exception:
                    pass

    total_time = time.time() - decode_start
    ttft_sec = (first_token_time - decode_start) if first_token_time else 0
    decode_time = total_time - ttft_sec
    decode_speed = token_count / decode_time if decode_time > 0 else 0

    print(f"\n  TTFT: {ttft_sec*1000:.1f} ms")
    print(f"  Decode 时间: {decode_time:.2f} s")
    print(f"  生成 tokens: {token_count}")
    print(f"  Decode 速度: {decode_speed:.1f} tok/s")

    return {
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "compression_ratio": compression_ratio,
        "prefill_time": prefill_time,
        "prefill_speed": original_tokens / prefill_time,
        "ttft_ms": ttft_sec * 1000,
        "decode_time": decode_time,
        "decode_speed": decode_speed,
        "token_count": token_count
    }

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    target_size = int(sys.argv[2]) if len(sys.argv) > 2 else 262144  # 默认 256K

    print("="*70)
    print("  PFlash + DFlash 精确性能测试")
    print("="*70)
    print(f"  服务地址: {url}")
    print(f"  目标 context: {target_size} tokens ({target_size//1024}K)")

    # 加载 tokenizer
    print("\n加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

    # 等待服务就绪
    print("\n等待服务就绪...")
    for i in range(60):
        try:
            resp = requests.get(f"{url}/v1/models", timeout=5)
            if resp.status_code == 200:
                print("✅ 服务就绪")
                break
        except Exception:
            pass
        time.sleep(1)

    # 创建 prompt
    print(f"\n创建 {target_size} tokens prompt...")
    prompt = create_long_prompt(target_size, tokenizer)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"  实际 tokens: {actual_tokens}")

    # 测试
    result = test_precise(url, prompt, tokenizer, label=f"测试 {target_size//1024}K Context")

    if result:
        print("\n" + "="*70)
        print("  测试结果")
        print("="*70)
        print(f"  原始 tokens: {result['original_tokens']}")
        print(f"  压缩后 tokens: {result['compressed_tokens']}")
        print(f"  压缩比: {result['compression_ratio']:.1f}:1")
        print(f"  Prefill 时间: {result['prefill_time']:.2f} s")
        print(f"  Prefill 速度: {result['prefill_speed']:.1f} tok/s")
        print(f"  TTFT: {result['ttft_ms']:.0f} ms")
        print(f"  Decode 速度: {result['decode_speed']:.1f} tok/s")
        print(f"  1024 tokens 用时: {1024 / result['prefill_speed']:.2f} s")
        print("="*70)

if __name__ == "__main__":
    main()
