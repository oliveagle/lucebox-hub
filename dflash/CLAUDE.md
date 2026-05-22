# DFlash TriAttention 项目

## 模型文件位置

**所有测试和推理使用以下模型对**，不要假设模型不存在：

| 角色 | 路径 (相对项目根目录) | 大小 | 量化 |
|------|------|------|------|
| Target | `models/Qwen3.6-27B-Q4_K_M.gguf` | ~15 GB | Q4_K_M |
| Draft | `models/draft/dflash-draft-3.6-q8_0.gguf` | ~1.7 GB | Q8_0 |

### 关键路径

- 项目根目录: `/mnt/eaget-4tb/data/llm_server/lucebox-hub-gfx1151/dflash/`
- 构建目录: `dflash/build/`
- 模型目录: `dflash/models/`

### 测试命令示例

```bash
cd dflash/build

# 创建测试 prompt
python3 -c "
import struct
tokens = [151644, 872, 570]
with open('/tmp/test_prompt.bin', 'wb') as f:
    for t in tokens:
        f.write(struct.pack('<I', t))
"

# 运行 DFlash 推理
./test_dflash \
  ../models/Qwen3.6-27B-Q4_K_M.gguf \
  ../models/draft/dflash-draft-3.6-q8_0.gguf \
  /tmp/test_prompt.bin \
  50 \
  /tmp/test_output.bin

# 基线测试 (AR)
./test_generate ../models/Qwen3.6-27B-Q4_K_M.gguf /tmp/test_prompt.bin 50 /tmp/test_output.bin
```

## TriAttention 配置

通过环境变量启用 TriAttention KV 压缩：

```bash
export TRIATTN_ENABLED=1
export TRIATTN_STATS_PATH=deps/llama.cpp/triattention/stats/qwen3.6-27b.bin
export TRIATTN_KV_BUDGET=512       # Max tokens to retain (默认 512)
export TRIATTN_DIVIDE_LENGTH=128   # Compression trigger interval
export TRIATTN_WINDOW_SIZE=128     # Recent tokens always preserved
export TRIATTN_MIN_KEEP_RATIO=0.5  # Minimum keep ratio (default 50%)
export TRIATTN_FORCE_COMPRESS=1    # Force compress every divide_length (for testing/debug)
```

**关键配置说明**：

- `TRIATTN_KV_BUDGET`: 压缩后保留的最大 token 数。默认值从 2048 改为 512，以更容易触发压缩
- `TRIATTN_FORCE_COMPRESS`: 设置为 1 时，无论 budget 如何都会每 `divide_length` 个 token 压缩一次（用于测试/调试）
- 压缩触发条件: `divide_length` 已过 AND (`committed > kv_budget` OR `force_compress=true`)
- **短上下文测试**: 使用 `TRIATTN_FORCE_COMPRESS=1` 来验证压缩功能是否正常

Stats 文件路径: `dflash/deps/llama.cpp/triattention/stats/qwen3.6-27b.bin`

## GPU 信息

- GPU: Radeon 8060S Graphics (gfx1151)
- 总显存: 128 GB (126976 MiB)
- 后端: ROCm/HIP (GGML_USE_HIP)

## 编译

```bash
cd dflash/build
cmake --build . --target test_dflash test_generate -j8
```

CMake 选项: `-DDFLASH27B_TRIATTENTION=ON -DGGML_TRIATTENTION=ON`
