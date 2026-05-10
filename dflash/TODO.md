# DFlash C++ HTTP 服务器 TODO

## 当前状态

✅ **已完成**: C++ HTTP 服务器框架已搭建，编译成功

## 待实现功能 (按优先级排序)

### 优先级 1: 核心功能 (必须实现)

- [ ] **Tokenizer 集成**
  - [x] 创建 src/server/tokenizer.h
  - [x] 创建 src/server/tokenizer.cpp
  - [ ] 创建 src/server/tokenizer_parser.h (辅助工具)
  - [ ] 使用 llama.cpp 的 gguf_get_tokenizer() 加载分词器
  - [ ] 实现 encode() 方法：文本 → token IDs
  - [ ] 实现 decode() 方法：token IDs → 文本
  - [ ] 实现 vocab_size() 属性

- [ ] **DFlash 解码循环实现**
  - [ ] 从 test/test_dflash.cpp 提取 prefill 逻辑
  - [ ] 从 test/test_dflash.cpp 提取 decode 循环逻辑
  - [ ] 实现到 DFlashServer::generate_internal() 方法
  - [ ] 支持 DFlash + DDTree 推测解码
  - [ ] 支持流式 token 回调

- [ ] **Chat Template 引擎**
  - [ ] 创建 src/server/chat_template.h
  - [ ] 创建 src/server/chat_template.cpp
  - [ ] 使用 llama.cpp 的 Jinja2 C++ 运行时
  - [ ] 实现消息列表 → prompt tokens 转换
  - [ ] 支持 add_generation_prompt=True
  - [ ] 支持 tools 参数

### 优先级 2: 高级功能 (重要但非阻塞)

- [ ] **工具调用解析**
  - [ ] 创建 src/server/parsers.h
  - [ ] 创建 src/server/parsers.cpp
  - [ ] 实现正则表达式解析 Qwen3 工具调用格式
  - [ ] 转换为 OpenAI 格式响应
  - [ ] 支持参数类型转换 (string/int/float/bool)

- [ ] **Thinking 模式解析**
  - [ ] 实现 parse_reasoning() 函数
  - [ ] 提取 reasoning 和 content 分离
  - [ ] 支持 chat_template_kwargs 控制

- [ ] **流式响应 (SSE)**
  - [ ] 实现 text/event-stream 格式
  - [ ] 实时 token 推送
  - [ ] 支持 stream_options.include_usage

- [ ] **Prefix Cache**
  - [ ] 实现 KV cache 快照
  - [ ] 支持 cache 命中和恢复
  - [ ] 实现 LRU 淘汰策略

- [ ] **PFlash 压缩**
  - [ ] 集成 PFlash drafter
  - [ ] 实现压缩协调逻辑
  - [ ] 支持 off/auto/always 模式

### 优先级 3: 完善和优化

- [ ] **Anthropic Messages API**
  - [ ] 实现 /v1/messages 端点
  - [ ] 支持 Anthropic 流式格式

- [ ] **日志和监控**
  - [ ] 实现请求日志
  - [ ] 实现性能指标
  - [ ] 添加错误追踪

- [ ] **错误处理**
  - [ ] 完善异常捕获
  - [ ] 添加重试逻辑
  - [ ] 实现优雅关闭

## 构建和运行

### 构建
```bash
cmake -B build -DDFLASH27B_HTTP_SERVER=ON
cmake --build build --target dflash_server -j
```

### 运行
```bash
./build/dflash_server \
    --target models/Qwen3.6-27B-Q4_K_M.gguf \
    --draft models/draft/model.safetensors \
    --prefill-always
```

### 测试
```bash
./scripts/test_server.sh
```

## 文件清单

### 新建文件
- src/server/dflash_server.h - HTTP 服务器头文件
- src/server/dflash_server.cpp - HTTP 服务器实现
- src/server/tokenizer.h - Tokenizer 封装 (待创建)
- src/server/tokenizer.cpp - Tokenizer 实现 (待创建)
- src/server/chat_template.h - Chat template 引擎 (待创建)
- src/server/chat_template.cpp - Chat template 实现 (待创建)
- src/server/parsers.h - 工具调用/Thinking 解析 (待创建)
- src/server/parsers.cpp - 解析器实现 (待创建)
- scripts/test_server.sh - 测试脚本

### 修改文件
- CMakeLists.txt - 添加 HTTP 服务器构建目标
- scripts/start_server.sh - 添加 C++ 服务器选项

## 参考
- src/server/README.md - 详细文档
- scripts/server_tools.py - Python 版本参考
