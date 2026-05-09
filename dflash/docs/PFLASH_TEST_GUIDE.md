# PFlash Speculative Prefill 测试指南

## 问题说明

当直接运行 `test_dflash --daemon` 时，程序会等待 stdin 输入，所以看起来"没有任何信息"就结束了。

## 正确的使用方法

### 方法 1: 使用 echo 管道命令

```bash
# 启动 daemon 并发送 park 命令（释放内存）
echo "park" | DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
    ./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/model.safetensors --daemon
```

### 方法 2: 使用交互式 Shell

```bash
# 1. 启动 daemon
DFLASH_FP_USE_BSA=1 DFLASH_FP_ALPHA=0.85 \
    ./build/test_dflash models/Qwen3.6-27B-Q4_K_M.gguf models/draft/model.safetensors --daemon

# 2. 在另一个终端，找到进程 ID 并发送命令
# 首先发送 park 命令释放内存
echo "park" > /proc/<pid>/fd/0

# 然后发送 compress 命令
echo "compress /tmp/test_prompt.bin 100 models/Qwen3-0.6B-BF16.gguf" > /proc/<pid>/fd/0
```

### 方法 3: 使用 Python 脚本（推荐）

```python
#!/usr/bin/env python3
import subprocess
import struct

# 启动 daemon
proc = subprocess.Popen(
    ["./build/test_dflash", "models/Qwen3.6-27B-Q4_K_M.gguf", "models/draft/model.safetensors", "--daemon"],
    env={"DFLASH_FP_USE_BSA": "1", "DFLASH_FP_ALPHA": "0.85", **os.environ},
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=False
)

# 1. Park 模型释放内存
proc.stdin.write(b"park\n")
proc.stdin.flush()

# 读取确认 (-1)
ack = struct.unpack("<i", proc.stdout.read(4))[0]
assert ack == -1

# 2. 发送 compress 命令
proc.stdin.write(b"compress /tmp/test_prompt.bin 100 models/Qwen3-0.6B-BF16.gguf\n")
proc.stdin.flush()

# 3. 读取压缩结果
compressed_ids = []
while True:
    tok_id = struct.unpack("<i", proc.stdout.read(4))[0]
    if tok_id == -1:
        break
    compressed_ids.append(tok_id)

print(f"Compressed {len(compressed_ids)} tokens")

# 清理
proc.stdin.write(b"exit\n")
proc.terminate()
```

### 方法 4: 使用现有的 server.py（最简单）

```bash
# server.py 内置了 PFlash 支持
python3 scripts/server.py \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --port 8000 \
    --daemon \
    --prefill-mode pflash \
    --prefill-threshold 1024
```

## 预期输出

当 PFlash 正常工作时，你会看到：

```
[compress] target parked
[compress] draft parked
[drafter] loaded models/Qwen3-0.6B-BF16.gguf (n_layer=24 n_head=12 n_head_kv=2)
[compress] 10000 -> 1200 tokens (keep_ratio=0.120)
[compress] target restored
[compress] draft restored
```

## 关键参数

- `DFLASH_FP_USE_BSA=1`: 启用 Block-Sparse Attention 加速
- `DFLASH_FP_ALPHA=0.85`: 压缩阈值（保留 85% 的重要性分数）
- `keep_x1000`: 保留比例 × 1000（如 100 = 0.10 = 保留 10%）
- `DFLASH_COMPRESS_NO_PARK=1`: 32GB+ VRAM 时可以不 park 模型

## 故障排查

### 问题 1: "out of memory"
- 解决：在 compress 之前先发送 `park` 命令
- 原因：daemon 模式默认会加载 target + draft，约占用 20GB

### 问题 2: daemon 无响应
- 检查：确认命令格式正确，特别是文件路径
- 调试：查看 stderr 输出获取错误信息

### 问题 3: 压缩比太低
- 调整：降低 `keep_x1000` 值（如从 200 降到 100）
- 注意：过低的保留比例可能影响生成质量
