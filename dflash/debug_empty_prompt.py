#!/usr/bin/env python3
"""Debug script to reproduce empty prompt issue"""
import json
import struct
import tempfile
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from transformers import AutoTokenizer

# Create 30 dummy tools
tools = []
for i in range(30):
    tools.append({
        "type": "function",
        "function": {
            "name": f"tool_{i}",
            "description": f"Tool number {i}",
            "parameters": {
                "type": "object",
                "properties": {
                    "arg1": {"type": "string", "description": f"Argument {i}"},
                },
                "required": ["arg1"]
            }
        }
    })

# Create a long user message
long_content = "Test " * 10000  # 50K chars

messages = [
    {"role": "user", "content": long_content}
]

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-27B", trust_remote_code=True)

# Convert to format expected by chat template
msgs_list = []
for m in messages:
    d = {"role": m["role"], "content": m["content"]}
    msgs_list.append(d)

# Apply chat template with tools
kwargs = dict(tokenize=False, add_generation_prompt=True)
tools_arg = [t["function"] | {"type": t["type"]} for t in tools]
kwargs["tools"] = tools_arg

print(f"Testing with {len(tools)} tools and user message length: {len(long_content)} chars")
print("=" * 60)

# Apply template
prompt = tokenizer.apply_chat_template(msgs_list, **kwargs)

print(f"Template applied. Prompt length: {len(prompt)} chars")
print(f"First 200 chars: {prompt[:200]}")
print(f"Last 200 chars: {prompt[-200:]}")
print()

# Encode
ids = tokenizer.encode(prompt, add_special_tokens=False)
print(f"Encoded to {len(ids)} tokens")
print()

# Write to temp file
fd, path = tempfile.mkstemp(suffix=".bin")
tmp_path = Path(path)

with open(fd, "wb") as f:
    for t in ids:
        f.write(struct.pack("<i", int(t)))

print(f"Written to: {tmp_path}")
print(f"File size: {tmp_path.stat().st_size} bytes ({tmp_path.stat().st_size // 4} tokens)")

# Read back and verify
with open(tmp_path, "rb") as f:
    data = f.read()
    read_ids = [struct.unpack_from("<i", data, i)[0] for i in range(0, len(data), 4)]
    print(f"Read back {len(read_ids)} tokens")
    print(f"First 10 tokens: {read_ids[:10]}")
    print(f"Last 10 tokens: {read_ids[-10:]}")

# Check if prompt is empty after template
if not prompt or prompt.isspace():
    print("\n⚠️  WARNING: Prompt template returned empty or whitespace!")
else:
    print(f"\n✓ Prompt is non-empty: {len(prompt)} chars")

# Clean up
tmp_path.unlink()

print("\n" + "=" * 60)
print("Test completed successfully!")
