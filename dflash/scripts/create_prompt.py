#!/usr/bin/env python3
"""
Create test prompts quickly
"""
import struct
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", trust_remote_code=True)

for size in [4096, 32768, 262144]:
    base_text = "深度学习是机器学习的一个分支，它是一种基于人工神经网络的学习方法。注意力机制允许模型动态关注输入的不同部分。"
    base_ids = tokenizer.encode(base_text, add_special_tokens=False)
    repeat = (size // len(base_ids)) + 1
    test_ids = (base_ids * repeat)[:size]

    output_file = f"/tmp/test_prompt_{size//1024}k.bin"
    with open(output_file, "wb") as f:
        for t in test_ids:
            f.write(struct.pack("<i", int(t)))

    print(f"Created {output_file} with {len(test_ids)} tokens")
