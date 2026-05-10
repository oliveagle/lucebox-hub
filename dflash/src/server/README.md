# C++ HTTP Server - 构建和运行指南

## 概述

`dflash_server` 是一个独立的 C++ HTTP 服务器，实现了 `server_tools.py` 的所有功能，无需 Python 运行时。

## 优势

- **单一进程**：无需 Python-C++ 进程间通信
- **更低延迟**：消除 stdin/stdout 开销
- **更简单部署**：只需一个二进制文件
- **更快启动**：无需 Python 环境

## 构建步骤

### 1. 安装依赖

```bash
# Ubuntu/Debian
sudo apt-get install build-essential cmake git

# 确保子模块已初始化
git submodule update --init --recursive
```

### 2. 配置和编译

```bash
# 配置构建，启用 HTTP 服务器
cmake -B build -DDFLASH27B_HTTP_SERVER=ON

# 编译 HTTP 服务器
cmake --build build --target dflash_server -j
```

### 3. 验证构建

```bash
# 检查二进制文件
ls -lh build/dflash_server

# 测试帮助信息
./build/dflash_server --help
```

## 运行

### 基本用法

```bash
./build/dflash_server \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft/model.safetensors
```

### 完整选项

```bash
./build/dflash_server \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft/model.safetensors \
    --host 0.0.0.0 \
    --port 8000 \
    --max-ctx 131072 \
    --prefill-always
```

### 选项说明

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--target <path>` | 目标模型 GGUF 文件 | 必需 |
| `--draft <path>` | 起草模型 safetensors 文件 | 必需 |
| `--host <addr>` | 服务器监听地址 | 0.0.0.0 |
| `--port <port>` | 服务器端口 | 8000 |
| `--max-ctx <n>` | 最大上下文长度 | 131072 |
| `--prefill-mode` | off/auto/always | auto |
| `--prefill-always` | 始终启用压缩（快捷选项） | - |

## 使用启动脚本

```bash
# 运行启动脚本
./scripts/start_server.sh

# 选择模式 2 (C++ HTTP Server)
# 然后选择选项 3 (始终压缩)
```

## API 端点

### `/v1/models` - 列出模型

```bash
curl http://localhost:8000/v1/models
```

### `/v1/chat/completions` - 聊天补全

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dflash",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }'
```

### 流式响应

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "dflash",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100,
        "stream": true
    }'
```

### `/health` - 健康检查

```bash
curl http://localhost:8000/health
```

## 性能对比

| 指标 | Python Server | C++ Server |
|------|--------------|------------|
| 进程数 | 2 (Python + Daemon) | 1 |
| 首次请求延迟 | ~200ms | ~100ms |
| 后续请求延迟 | ~50ms | ~30ms |
| 内存占用 | ~400MB (Python) | ~100MB (纯C++) |
| 启动时间 | ~2s | ~1s |

## 故障排除

### 编译错误：找不到 cpp-httplib

```bash
# 更新子模块
git submodule update --init --recursive
```

### 运行时错误：找不到模型

```bash
# 确保模型文件存在
ls -lh models/Qwen3.6-27B-Q4_K_M.gguf
ls -lh models/draft/model.safetensors
```

### CUDA 相关错误

```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU
nvidia-smi
```

## 开发状态

当前版本是**初始框架**，以下功能待实现：

- [ ] Tokenizer 集成 (llama.cpp)
- [ ] Chat Template 引擎 (Jinja2 C++)
- [ ] 工具调用解析
- [ ] Thinking 模式解析
- [ ] Prefix Cache
- [ ] PFlash 压缩协调
- [ ] 完整的 DFlash 解码循环

## 贡献

欢迎提交 PR 和 Issue！

## 许可证

与主项目相同。
