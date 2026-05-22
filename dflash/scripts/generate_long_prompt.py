#!/usr/bin/env python3
"""Generate long context prompt (8192+ tokens) for TriAttention performance testing.

Usage: python3 generate_long_prompt.py <tokenizer_path> <output_bin>

Since we don't have a tokenizer, we'll use the known Qwen3.6 special tokens
and a repeating pattern to fill up 8192+ tokens.
"""

import struct
import sys

def main():
    output_path = sys.argv[1] if len(sys.argv) > 1 else '/tmp/long_prompt_8192.bin'
    target_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8192

    # Qwen3.6 special tokens (from known Qwen3.6 tokenizer)
    # <|im_start|> = 151644, user = 872, <|im_end|> = 151645
    # system = 9452, assistant = 76330, user = 872, etc.
    system_prefix = [151644, 9452]  # <|im_start|>system<|im_sep|>
    system_content = [74786, 99776]  # "You are a helpful assistant." (approximate tokens)
    system_suffix = [151645]  # <|im_end|>

    user_prefix = [151644, 872]  # <|im_start|>user<|im_sep|>
    user_suffix = [151645]  # <|im_end|>

    assistant_prefix = [151644, 76330]  # <|im_start|>assistant<|im_sep|>

    # Build a very long user message (repetitive content to fill 8192+ tokens)
    # Use a simple repeating paragraph
    # These are approximate Qwen token IDs for common English words
    # We'll use a safe approach: use a sequence of high-probability token IDs
    # from the Qwen3 vocabulary range (15000-50000 for most Chinese/English tokens)

    # A safer approach: use known common tokens
    # Let's use token IDs that are very likely to exist
    # Qwen3 vocab size is ~151,936. Most printable text tokens are in 300-100000 range.

    # Generate a repeating text pattern with plausible token IDs
    # Using a simple pattern: sequence of ~200 token IDs repeated
    # These are chosen from common token ranges to avoid special tokens

    # Common English word approximate token IDs (Qwen2.5/3 range)
    # Use a larger range to support 64K tokens
    text_tokens = list(range(1000, 5000))  # Safe range of regular tokens (4000 tokens)

    tokens = []

    # System message
    tokens.extend(system_prefix)
    tokens.extend(system_content)
    tokens.extend(system_suffix)

    # User message with long context
    tokens.extend(user_prefix)

    # Fill up to target_tokens - overhead for prefixes/suffixes
    overhead = len(tokens) + len(user_suffix) + len(assistant_prefix) + 10  # margin
    fill_needed = target_tokens - overhead

    repeat_count = fill_needed // len(text_tokens) + 1
    expanded = (text_tokens * repeat_count)[:fill_needed]
    tokens.extend(expanded)

    tokens.extend(user_suffix)

    # Assistant prefix (so model starts generating)
    tokens.extend(assistant_prefix)

    # Write binary
    with open(output_path, 'wb') as f:
        for t in tokens:
            f.write(struct.pack('<I', t))

    print(f"Generated {len(tokens)} tokens to {output_path}")

if __name__ == '__main__':
    main()
