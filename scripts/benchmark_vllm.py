#!/usr/bin/env python3
"""
vLLM Baseline Benchmark Script

Measures performance metrics for vLLM without TriAttention compression:
- Prefill speed (tokens/sec)
- Decode speed (tokens/sec)
- First token latency (ms)

Usage:
    python benchmark_vllm.py --model Qwen/Qwen3.6-27B-AWQ --base-url http://localhost:8000
"""

import argparse
import json
import time
from datetime import datetime
from openai import OpenAI


def run_benchmark(model_name: str, base_url: str, context_length: int, output_length: int) -> dict:
    """
    Run a single benchmark scenario.

    Args:
        model_name: HuggingFace model name
        base_url: vLLM server URL
        context_length: Input prompt length (tokens)
        output_length: Expected output length (tokens)
    """
    client = OpenAI(api_key="token", base_url=f"{base_url}/v1")

    # Generate test prompt
    prompt = "Write a detailed Python function that implements quicksort. "
    # Repeat to reach approximate context length
    prompt = prompt * max(1, context_length // 20)

    print(f"\n{'='*60}")
    print(f"Test Scenario:")
    print(f"  Context Length: {context_length} tokens")
    print(f"  Output Length: {output_length} tokens")
    print(f"  Actual Prompt: {len(prompt)} chars")
    print(f"{'='*60}")

    # Run the generation
    start_time = time.time()

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=output_length,
        stream=True,
    )

    # Process streaming response to measure first token latency
    first_token_time = None
    tokens_generated = 0

    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            if first_token_time is None:
                first_token_time = time.time() - start_time
                print(f"First token latency: {first_token_time * 1000:.2f} ms")
            tokens_generated += 1

    total_time = time.time() - start_time
    decode_time = total_time - first_token_time if first_token_time else total_time

    # Calculate metrics
    prefill_speed = context_length / (first_token_time or 1)  # tokens/sec
    decode_speed = tokens_generated / decode_time if decode_time > 0 else 0  # tokens/sec

    print(f"\nResults:")
    print(f"  Prefill Speed: {prefill_speed:.2f} tokens/sec")
    print(f"  Decode Speed: {decode_speed:.2f} tokens/sec")
    print(f"  First Token Latency: {(first_token_time or 0) * 1000:.2f} ms")
    print(f"  Total Time: {total_time:.2f} sec")
    print(f"  Tokens Generated: {tokens_generated}")
    print(f"{'='*60}")

    return {
        "context_length": context_length,
        "output_length": output_length,
        "prefill_tokens": context_length,
        "prefill_time": first_token_time,
        "prefill_speed": prefill_speed,
        "decode_tokens": tokens_generated,
        "decode_time": decode_time,
        "decode_speed": decode_speed,
        "first_token_latency_ms": (first_token_time or 0) * 1000,
        "total_time": total_time,
        "tokens_generated": tokens_generated,
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM Baseline Benchmark")
    parser.add_argument("--model", required=True, help="Model name (e.g., Qwen/Qwen3.6-27B-AWQ)")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--scenarios", nargs="*", type=int, default=[(4096, 512), (8192, 512), (16384, 512)])
    parser.add_argument("--output-file")
    args = parser.parse_args()

    # Parse scenarios as pairs if provided as flat list
    scenarios = []
    for i in range(0, len(args.scenarios), 2):
        scenarios.append((args.scenarios[i], args.scenarios[i+1]))

    if not scenarios:
        scenarios = [(4096, 512), (8192, 512), (16384, 512)]

    print(f"\nvLLM Baseline Benchmark")
    print(f"Model: {args.model}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Base URL: {args.base_url}")

    results = []

    for ctx_len, out_len in scenarios:
        result = run_benchmark(args.model, args.base_url, ctx_len, out_len)
        results.append(result)

    # Output summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Context':>10} | {'Decode':>12} | {'Prefill':>10} | {'FTL':>10}")
    print(f"{'Length':>10} | {'(tok/s)':>12} | {'(tok/s)':>10} | {'(ms)':>10}")
    print(f"-" * 60)

    for r in results:
        print(f"{r['context_length']:>10} | {r['decode_speed']:>12.2f} | {r['prefill_speed']:>10.2f} | {r['first_token_latency_ms']:>10.2f}")

    print(f"{'='*60}")

    # Save to file
    output_data = {
        "engine": "vLLM (Baseline, no TriAttention)",
        "model": args.model,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "results": results,
    }

    output_path = args.output_file or f"baseline_vllm_{datetime.now().strftime('%Y%m%d')}.json"

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_data


if __name__ == "__main__":
    main()
