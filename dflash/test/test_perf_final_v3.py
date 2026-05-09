#!/usr/bin/env python3
"""
PFlash + DFlash 完整性能测试
测量 Prefill（首包时间）和 Decode（token/s）性能
"""
import requests
import json
import time
from dataclasses import dataclass
from typing import List
import sys


@dataclass
class PerfResult:
    """性能测试结果"""
    name: str
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_time: float
    prefill_tok_s: float
    decode_tok_s: float
    total_tok_s: float
    compression_ratio: float = 1.0


def test_request(
    url: str,
    prompt: str,
    max_tokens: int = 128,
    description: str = "Test",
    disable_thinking: bool = True
) -> PerfResult:
    """测试单个请求"""
    print(f"\n{'='*70}")
    print(f"  {description}")
    print(f"{'='*70}")

    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    start = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, stream=True, timeout=600)

    if resp.status_code != 200:
        print(f"❌ 错误: {resp.status_code} - {resp.text}")
        return None

    first_token_time = None
    token_count = 0
    prompt_tokens_est = 0

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
                    delta = data.get('choices', [{}])[0].get('delta', {})
                    if delta.get('content'):
                        token_count += 1
                except Exception:
                    pass

    total_time = time.time() - start
    ttft = (first_token_time - start) * 1000 if first_token_time else 0

    # 从第二个请求获取 prompt tokens（非流式
    payload2 = payload.copy()
    payload2['stream'] = False
    payload2['max_tokens'] = 1
    resp2 = requests.post(f"{url}/v1/chat/completions", json=payload2, timeout=600)
    if resp2.status_code == 200:
        result = resp2.json()
        prompt_tokens_est = result['usage']['prompt_tokens']
        original_prompt_tokens = result['usage']['prompt_tokens']
    else:
        prompt_tokens_est = 0
        original_prompt_tokens = 0

    # 获取 completion tokens（从流式响应）
    completion_tokens = token_count

    # 计算性能指标
    decode_time = total_time - (ttft / 1000) if ttft > 0 else total_time
    prefill_time = ttft / 1000

    prefill_tok_s = prompt_tokens_est / prefill_time if prefill_time > 0 else 0
    decode_tok_s = completion_tokens / decode_time if decode_time > 0 else 0
    total_tok_s = (prompt_tokens_est + completion_tokens) / total_time if total_time > 0 else 0

    print(f"  Prompt: {prompt_tokens_est} tokens")
    print(f"  TTFT: {ttft:.1f} ms")
    print(f"  Completion: {completion_tokens} tokens")
    print(f"  Total time: {total_time:.2f} s")
    print(f"  Prefill: {prefill_tok_s:.0f} tok/s")
    print(f"  Decode: {decode_tok_s:.1f} tok/s")
    print(f"  Total: {total_tok_s:.1f} tok/s")

    return PerfResult(
        name=description,
        prompt_tokens=prompt_tokens_est,
        completion_tokens=completion_tokens,
        ttft_ms=ttft,
        total_time=total_time,
        prefill_tok_s=prefill_tok_s,
        decode_tok_s=decode_tok_s,
        total_tok_s=total_tok_s
    )


def run_comprehensive_test(url: str, use_long_prompts: bool = False) -> List[PerfResult]:
    """运行完整性能测试"""
    results = []

    print("\n" + "="*70)
    print("  PFlash + DFlash 性能测试套件")
    print("="*70)

    if use_long_prompts:
        from transformers import AutoTokenizer
        print("\n加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

        def create_prompt(tokens):
            base = "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。"
            repeat = (tokens // 20) + 2
            full = base * repeat
            ids = tokenizer.encode(full, add_special_tokens=False)[:tokens]
            return tokenizer.decode(ids, skip_special_tokens=False)

        # 测试场景
        test_cases = [
            ("2K tokens", 2048, 64),
            ("4K tokens", 4096, 64),
            ("8K tokens", 8192, 64),
            ("16K tokens (PFlash)", 16384, 64),
        ]

        for name, prompt_len, max_tokens in test_cases:
            prompt = create_prompt(prompt_len)
            result = test_request(url, prompt, max_tokens, name)
            if result:
                results.append(result)
            time.sleep(2)
    else:
        # 简单测试
        test_cases = [
            ("Short (你好)", "你好，请介绍一下你自己。", 128),
            ("Medium (深度学习)", "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。深度学习能够通过多层神经网络自动学习特征表示，广泛应用于计算机视觉、自然语言处理、语音识别等领域。", 128),
        ]

        for name, prompt, max_tokens in test_cases:
            result = test_request(url, prompt, max_tokens, name)
            if result:
                results.append(result)
            time.sleep(1)

    return results


def print_summary(results: List[PerfResult]):
    """打印测试结果摘要"""
    print("\n" + "="*70)
    print("  测试结果摘要")
    print("="*70)

    print(f"\n{'场景':<25} {'Prompt':<10} {'TTFT(ms)':<12} {'Prefill':<12} {'Decode':<12} {'Total':<12}")
    print("-"*90)

    for r in results:
        print(f"{r.name:<25} {r.prompt_tokens:<10} {r.ttft_ms:<12.1f} {r.prefill_tok_s:<12.0f} {r.decode_tok_s:<12.1f} {r.total_tok_s:<12.1f}")

    print("\n" + "="*70)
    print("  性能分析")
    print("="*70)

    if results:
        avg_decode = sum(r.decode_tok_s for r in results) / len(results)
        avg_prefill = sum(r.prefill_tok_s for r in results if r.prefill_tok_s > 0) / len([r for r in results if r.prefill_tok_s > 0])
        avg_ttft = sum(r.ttft_ms for r in results) / len(results)

        print(f"\n  平均 TTFT: {avg_ttft:.1f} ms")
        print(f"  平均 Prefill 速度: {avg_prefill:.0f} tok/s")
        print(f"  平均 Decode 速度: {avg_decode:.1f} tok/s")


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    mode = sys.argv[2] if len(sys.argv) > 2 else "simple"

    print("="*70)
    print("  PFlash + DFlash 性能测试")
    print("="*70)
    print(f"  服务地址: {url}")
    print(f"  测试模式: {mode}")

    # 等待服务就绪
    print("\n等待服务就绪...")
    for i in range(90):
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
        print("⚠️  服务未就绪，但继续尝试...")

    if mode == "full":
        results = run_comprehensive_test(url, use_long_prompts=True)
    else:
        results = run_comprehensive_test(url, use_long_prompts=False)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
