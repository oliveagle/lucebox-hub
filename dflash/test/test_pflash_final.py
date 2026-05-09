#!/usr/bin/env python3
"""
PFlash + DFlash OpenAI API 测试（禁用思考
"""
import requests
import json
import time
import sys


def test_no_thinking_disabled(base_url="http://localhost:8000"):
    """测试禁用思考后的响应"""
    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": "你好，请介绍一下深度学习"}],
        "max_tokens": 100,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    print("="*70)
    print("  测试禁用思考 (enable_thinking=False")
    print("="*70)
    print("="*70)

    start = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    ttft = time.time() - start

    print(f"  状态码: {resp.status_code}")
    print(f"  总时间: {ttft:.2f} s")

    if resp.status_code == 200:
        result = resp.json()
        print("="*70)
        print(f"  响应内容:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        usage = result.get('usage', {})

        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

        print(f"  prompt_tokens: {prompt_tokens}")
        print(f"  completion_tokens: {completion_tokens}")
        print(f"  total_tokens: {total_tokens}")
        print(f"  速度: {total_tokens / ttft:.2f} tok/s")

        msg = result.get('choices', [{}])[0].get('message', {})
        print("\n  响应消息内容:")
        if msg.get('reasoning_content'):
            print(f"  思考内容: {msg.get('reasoning_content')[:200]}...")
        if msg.get('content'):
            print(f"  文本内容: {msg.get('content')}")
    else:
        print(f"  错误: {resp.text}")

    print("="*70)


def test_stream_no_thinking(base_url="http://localhost:8000"):
    """测试禁用思考的流式响应"""
    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}],
        "max_tokens": 100,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    print("\n" + "="*70)
    print("  测试禁用思考的流式响应")
    print("="*70)

    start = time.time()
    ttft = None
    token_count = 0
    full_content = ""

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, stream=True, timeout=300)

    print(f"  响应状态码: {resp.status_code}")
    print("\n  流式输出:")

    if resp.status_code == 200:
        for line in resp.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)

                        if ttft is None:
                            ttft = time.time()
                        delta = data.get('choices', [{}])[0].get('delta', {})

                        # 优先获取 content，其次 reasoning_content
                        if 'content' in delta:
                            content = delta['content']
                            if content:
                                full_content += content
                                token_count += 1
                                print(content, end='', flush=True)
                        elif 'reasoning_content' in delta:
                            r_content = delta['reasoning_content']
                            if r_content:
                                full_content += r_content
                                token_count += 1
                                print(r_content, end='', flush=True)
                    except Exception:
                        pass

        total_time = time.time() - start
        ttft_sec = ttft - start if ttft else 0
        decode_time = total_time - ttft_sec if ttft else total_time

        print("\n\n" + "="*70)
        print(f"  首包时间 (TTFT): {ttft_sec*1000:.1f} ms")
        print(f"  总时间: {total_time:.2f} s")
        print(f"  Token 数: {token_count}")
        print(f"  解码速度: {token_count / decode_time:.1f} tok/s")
        print(f"  总体速度: {token_count / total_time:.1f} tok/s")


def test_pflash_effect(base_url="http://localhost:8000"):
    """测试 PFlash 压缩效果"""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

    print("\n\n" + "="*70)
    print("  PFlash 压缩效果测试")
    print("="*70)

    def create_long_prompt(num_tokens):
        base = "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。"
        repeat = (num_tokens // 20) + 2
        full = base * repeat
        ids = tokenizer.encode(full, add_special_tokens=False)[:num_tokens]
        return tokenizer.decode(ids, skip_special_tokens=False)

    print("\n  1. 测试 8K tokens (不触发 PFlash)")
    prompt_8k = create_long_prompt(8000)
    prompt_8k_len = len(tokenizer.encode(prompt_8k, add_special_tokens=False))
    print(f"  实际 prompt tokens: {prompt_8k_len}")

    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt_8k}],
        "max_tokens": 32,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    start = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    time_8k = time.time() - start
    print(f"  响应时间: {time_8k:.2f} s")

    if resp.status_code == 200:
        result = resp.json()
        print(f"  实际处理 prompt tokens: {result['usage']['prompt_tokens']}")

    print("\n  2. 测试 16K tokens (触发 PFlash)")
    prompt_16k = create_long_prompt(16000)
    prompt_16k_len = len(tokenizer.encode(prompt_16k, add_special_tokens=False))
    print(f"  实际 prompt tokens: {prompt_16k_len}")

    payload['messages'][0]['content'] = prompt_16k
    payload['max_tokens'] = 32

    start = time.time()
    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=300)
    time_16k = time.time() - start
    print(f"  响应时间: {time_16k:.2f} s")

    if resp.status_code == 200:
        result = resp.json()
        print(f"  实际处理 prompt tokens: {result['usage']['prompt_tokens']}")

        ratio = prompt_16k_len / result['usage']['prompt_tokens']
        print(f"\n  PFlash 压缩比: {ratio:.1f}x")
        print("="*70)


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"

    print("="*70)
    print("  PFlash + DFlash OpenAI API 测试")
    print("="*70)

    test_no_thinking_disabled(base_url)
    test_stream_no_thinking(base_url)
    test_pflash_effect(base_url)
