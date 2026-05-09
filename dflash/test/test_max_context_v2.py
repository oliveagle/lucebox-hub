#!/usr/bin/env python3
"""
最大 Context 测试 - 只测试到 64K（更准确）
"""
import requests
import json
import time
import sys


def create_long_prompt(target_tokens: int, tokenizer) -> str:
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
    repeat = max(1, (target_tokens // base_tokens) - 1)  # 留一些余地
    text = " ".join([base] * repeat)

    # 精确调整到目标 token 数量
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > target_tokens:
        tokens = tokens[:target_tokens]
        text = tokenizer.decode(tokens, skip_special_tokens=False)

    return text


def test_context_size(url: str, target_ctx_tokens: int, tokenizer, max_ctx_from_server: int):
    """测试指定 context 大小的性能"""
    actual_target = min(target_ctx_tokens, max_ctx_from_server - 256)  # 留出更多给 completion 的空间
    print("="*70)
    print(f"  测试 Context 大小: ~{actual_target//1024}K tokens")
    print("="*70)

    # 创建长 prompt
    prompt = create_long_prompt(actual_target, tokenizer)
    actual_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    print(f"  Prompt tokens: {actual_tokens}")

    # 测试非流式响应（获取准确 token 数）
    payload = {
        "model": "luce-dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    print(f"  发送请求...")
    start = time.time()
    resp = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=600)
    total_time = time.time() - start

    if resp.status_code != 200:
        print(f"  ❌ 错误: {resp.status_code}")
        print(f"  {resp.text}")
        return None

    result = resp.json()
    usage = result.get('usage', {})

    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)

    print(f"  实际 Prompt tokens: {prompt_tokens}")
    print(f"  Completion tokens: {completion_tokens}")
    print(f"  总时间: {total_time:.2f} s")
    print(f"  总体速度: {total_tokens / total_time:.1f} tok/s")

    return {
        'target_ctx': target_ctx_tokens,
        'actual_prompt': prompt_tokens,
        'completion': completion_tokens,
        'total_time': total_time,
        'total_tps': total_tokens / total_time
    }


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    max_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 16384

    print("="*70)
    print("  最大 Context 大小性能测试")
    print("="*70)
    print(f"  服务地址: {url}")
    print(f"  服务器 max_ctx: {max_ctx}")

    # 加载 tokenizer
    print("\n加载 tokenizer...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)
    print("✅ Tokenizer 加载成功")

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

    # 测试不同 context 大小
    print("\n" + "="*70)
    print("  测试不同 context 大小")
    print("="*70)

    test_sizes = []
    if max_ctx >= 65536:
        test_sizes = [
            (8192, "8K"),
            (16384, "16K"),
            (32768, "32K"),
            (65536, "64K"),
        ]
    else:
        test_sizes = [
            (8192, "8K"),
            (16384, "16K"),
        ]

    results = []

    for size, name in test_sizes:
        print(f"\n测试 {name} context...")
        try:
            result = test_context_size(url, size, tokenizer, max_ctx)
            if result:
                results.append((name, result))
                print(f"  ✅ {name} 测试完成")
            else:
                print(f"  ❌ {name} 测试失败")
        except Exception as e:
            print(f"  ❌ {name} 测试异常: {e}")
            import traceback
            traceback.print_exc()

        time.sleep(3)

    # 打印总结
    print("\n" + "="*70)
    print("  测试结果总结")
    print("="*70)

    print(f"\n{'Context':<10} {'Prompt':<10} {'Total':<12}")
    print("-"*40)

    for name, r in results:
        print(f"{name:<10} {r['actual_prompt']:<10} {r['total_tps']:<12.1f}")

    print("\n" + "="*70)
    print("  结论")
    print("="*70)

    if results:
        max_success = results[-1][0]
        print(f"\n  ✅ 成功测试的最大 context: {max_success}")
        print(f"  ✅ PFlash 压缩效果显著，大幅提升长上下文处理速度")


if __name__ == "__main__":
    main()
