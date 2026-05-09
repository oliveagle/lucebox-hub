#!/usr/bin/env python3
import struct

# Create a small prompt file
with open('/tmp/test_small_prompt.bin', 'wb') as f:
    for i in range(10):
        f.write(struct.pack('<i', i))
