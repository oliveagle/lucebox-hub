#!/usr/bin/env python3
"""
精确的 decode 性能测试
"""
import requests
import json
import time
import sys


def test_decode_speed(url: str, max_tokens: int = 256, prompt_tokens: int = 30):
    """测试精确的 decode 速度"""
    print("="*70)
    print(f"  Decode 性能测试")
    print("="*70)

    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": "请详细介绍深度学习的原理和应用，包括卷积神经网络、循环神经网络和 Transformer 架构。"}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    print(f"  发送请求...")
    start = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, stream=True, timeout=600)

    if resp.status_code != 200:
        print(f"❌ 错误: {resp.status_code} - {resp.text}")
        return

    first_token_time = None
    token_count = 0
    content = ""

    print(f"  接收流式响应...\n")

    for line in resp.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data_str = line[6:]
                if data_str == '[DONE]':
                    break
                try:
                    data = json.loads(data_str)
                    if first_token_time is None:
                        first_token_time = time.time()
                        ttft = (first_token_time - start) * 1000
                        print(f"  [首包时间 TTFT: {ttft:.1f} ms]\n")

                    delta = data.get('choices', [{}])[0].get('delta', {})
                    if delta.get('content'):
                        text = delta['content']
                        content += text
                        token_count += 1
                        print(text, end='', flush=True)
                except Exception as e:
                    pass

    total_time = time.time() - start

    # 计算精确的性能指标
    ttft_sec = (first_token_time - start) if first_token_time else 0
    decode_time = total_time - ttft_sec

    print("\n\n" + "="*70)
    print("  精确性能统计")
    print("="*70)
    print(f"  总时间: {total_time:.2f} s")
    print(f"  TTFT: {ttft_sec*1000:.1f} ms")
    print(f"  Decode 时间: {decode_time:.2f} s")
    print(f"  生成 tokens: {token_count}")
    print(f"  实际 decode 速度: {token_count / decode_time:.1f} tok/s")

    # 获取 prompt tokens
    payload2 = payload.copy()
    payload2['stream'] = False
    payload2['max_tokens'] = 1
    resp2 = requests.post(f"{url}/v1/chat/completions", json=payload2, timeout=300)
    if resp2.status_code == 200:
        result = resp2.json()
        prompt_tokens = result['usage']['prompt_tokens']
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Prefill 速度: {prompt_tokens / ttft_sec:.0f} tok/s")

    print("="*70)


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 256

    print("="*70)
    print("  精确 Decode 性能测试")
    print("="*70)
    print(f"  服务地址: {url}")
    print(f"  最大 tokens: {max_tokens}")

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
        if i % 10 == 0 and i > 0:
            print(f"  等待中... ({i}s)")
    else:
        print("⚠️  服务未就绪，继续尝试...")

    test_decode_speed(url, max_tokens)


if __name__ == "__main__":
    main()
