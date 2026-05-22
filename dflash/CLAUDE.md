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
export TRIATTN_KV_BUDGET=2048
export TRIATTN_DIVIDE_LENGTH=128
export TRIATTN_WINDOW_SIZE=128
```

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
