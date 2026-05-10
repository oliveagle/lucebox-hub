# DFlash C++ HTTP 服务器完整实现 TODO

## 当前状态

✅ **已完成**: C++ HTTP 服务器框架已搭建，编译成功
📍 **可执行文件**: `build/dflash_server`
🔧 **已有核心组件**: DFlash 解码、DDTree、PFlash、TurboQuant KV Cache 均已实现（在 `test/test_dflash.cpp` 中）

---

## 项目架构分析

### 已有核心组件（位于 `src/` 和 `test/`）

| 组件 | 文件 | 状态 | 说明 |
|------|------|------|------|
| **目标模型加载** | `gguf_target_loader.cpp` | ✅ | Qwen3.5-27B GGUF 加载 |
| **草稿模型加载** | `safetensors_draft.cpp` | ✅ | DFlash safetensors 加载 |
| **目标前向图** | `qwen35_target_graph.cpp` | ✅ | Qwen3.5 混合注意力图 |
| **草稿前向图** | `qwen3_dflash_graph.cpp` | ✅ | DFlash 推测解码图 |
| **KV Cache** | `kv_cache.cpp` | ✅ | 含 TurboQuant 支持 |
| **KV 量化** | `kv_quant.cpp` | ✅ | Q4_0/Q8_0/TQ3_0 等 |
| **DFlash 解码** | `test_dflash.cpp` | ✅ | 完整的 spec-decode 循环 |
| **DDTree** | `test_dflash.cpp` | ✅ | 树形推测解码 |
| **PFlash 压缩** | `flashprefill.cpp` | ✅ | FlashPrefill 稀疏注意力 |
| **PFlash Drafter** | `qwen3_0p6b_*.cpp` | ✅ | Qwen3-0.6B 压缩模型 |
| **Sampler** | `sampler.cpp` | ✅ | 采样链 (temp/top_p/top_k) |
| **Tokenizer** | `tokenizer.cpp` | ✅ | llama.cpp tokenizer |
| **工具解析** | `parsers.cpp` | ✅ | Qwen3 工具调用解析 |

---

## 服务器集成工作清单

### 优先级 0: 修复占位符 (必须先完成)

#### 0.1 移除 dflash_server.cpp 中的占位符类
- [ ] 删除 `Tokenizer` 占位符类 (line 29-46)
- [ ] 删除 `ChatTemplate` 占位符类 (line 50-61)
- [ ] 使用 `src/server/tokenizer.cpp` 中的真实 Tokenizer
- [ ] 更新 `dflash_server.h` 中的前向声明

#### 0.2 实现模型加载
- [ ] 在 `initialize()` 中调用 `load_target_gguf()`
- [ ] 在 `initialize()` 中调用 `load_draft_safetensors()` 或 `load_draft_gguf()`
- [ ] 在 `initialize()` 中调用 `create_target_cache()`
- [ ] 解析 KV 类型 (`resolve_kv_types()`)
- [ ] 设置 TurboQuant 参数 (`kv_k_rotated`, `kv_k_type`, `kv_v_type`)

---

### 优先级 1: 核心生成循环

#### 1.1 实现 `generate_internal()` 方法
**参考**: `test/test_dflash.cpp` 中的完整 DFlash 循环

```cpp
// 需要实现的逻辑流程：
std::vector<int32_t> DFlashServer::generate_internal(
    const std::vector<int32_t>& prompt_tokens,
    int n_gen,
    std::function<void(int32_t)> on_token
) {
    // 1. Prefill: 单 token decode 捕获层特征
    //    - 调用 build_qwen35_graph() 设置 capture_layers=true
    //    - 填充 cache.target_feat

    // 2. Decode 循环 (直到 n_gen 或 EOS):
    //    a. 构建 noise block [last_tok, MASK*15]
    //    b. Draft forward → 16 个候选 token
    //    c. Snapshot SSM 状态
    //    d. Batch target verify (16 个 draft tokens)
    //    e. Greedy longest-prefix accept + 1 bonus token
    //    f. Restore SSM 状态
    //    g. Replay accepted tokens
    //    h. 更新 committed, last_tok
    //    i. 调用 on_token() 流式输出
}
```

**子任务**:
- [ ] 实现 prefill 阶段 (单 token decode + 层特征捕获)
- [ ] 实现构建 noise embedding (`build_noise_block`)
- [ ] 实现草稿前向 (`build_draft_graph` + `ggml_backend_graph_compute`)
- [ ] 实现 SSM 状态快照 (`snapshot_ssm_state`)
- [ ] 实现目标验证 (batch verify with causal mask)
- [ ] 实现 token 接受逻辑 (longest-prefix match)
- [ ] 实现 SSM 状态恢复 (`restore_ssm_state`)
- [ ] 实现 token replay
- [ ] 实现 EOS 检查 (`IS_EOS_TOK`)

#### 1.2 Chat Template 引擎
- [ ] 创建 `src/server/chat_template.h`
- [ ] 创建 `src/server/chat_template.cpp`
- [ ] 使用 llama.cpp 的 Jinja2 C++ 运行时
- [ ] 实现消息列表 → prompt tokens 转换
- [ ] 支持 `add_generation_prompt=True`
- [ ] 支持 Qwen3.5 chat template

---

### 优先级 2: PFlash 压缩集成

#### 2.1 PFlash 协调器
**已有组件**: `PrefillCoordinator` 占位符 (line 108-130)

- [ ] 实现压缩决策逻辑 (`should_compress`)
- [ ] 集成 `load_drafter()` 加载 Qwen3-0.6B 模型
- [ ] 调用 `drafter_score_and_compress()` 执行压缩
- [ ] 实现 compress → decode 迁移 (`migrate_prefill_cache`)

**参考**: `scripts/server_tools.py` 的 `compress_prefill()`

#### 2.2 PFlash 模式支持
- [ ] off: 禁用压缩
- [ ] auto: 根据提示长度自动决策
- [ ] always: 始终压缩
- [ ] 实现 `prefill_threshold` 和 `prefill_keep_ratio` 参数

---

### 优先级 3: TurboQuant KV Cache

#### 3.1 KV 类型解析和设置
**已有组件**: `kv_quant.cpp` 提供完整支持

- [ ] 在 `initialize()` 中调用 `resolve_kv_types()`
- [ ] 支持环境变量配置:
  - `DFLASH27B_KV_K` - K 类型 (f16/bf16/q4_0/q8_0/tq3_0)
  - `DFLASH27B_KV_V` - V 类型 (f16/bf16/q4_0/q8_0/tq3_0)
  - `DFLASH27B_KV_F16` - F16 快捷方式
  - `DFLASH27B_KV_Q4` - Q4_0 快捷方式
  - `DFLASH27B_KV_TQ3` - TurboQuant 快捷方式

#### 3.2 KV Cache 创建和管理
**已有组件**: `kv_cache.cpp` 提供完整支持

- [ ] 使用 `create_target_cache()` 创建缓存
- [ ] 设置 `kv_k_rotated=true` 启用 TurboQuant
- [ ] 支持 `max_verify_tokens` 参数 (DDTree 需要)
- [ ] 实现 cache 重置 (`reset_target_cache`)

#### 3.3 KV Cache 序列化
**已有组件**: `snapshot_target_cache()`, `restore_target_cache()`

- [ ] 实现 Prefix Cache 快照
- [ ] 实现 thin snapshot (增量快照)
- [ ] 实现 chain restore (厚底 + 薄层)

---

### 优先级 4: DDTree 树形推测解码

#### 4.1 DDTree 核心逻辑
**已有组件**: `test/test_dflash.cpp` 中已实现

- [ ] 提取 DDTree 相关代码到独立函数
- [ ] 实现 `extract_draft_topk()` (已在 test_dflash.cpp)
- [ ] 实现最佳优先搜索 (`best_first_search`)
- [ ] 实现树掩码构建 (`build_tree_mask`)
- [ ] 支持动态树深度 (`tree_budget` 参数)

#### 4.2 DDTree 与服务器集成
- [ ] 添加 `--tree-budget` 命令行参数
- [ ] 在 `generate_internal()` 中支持 DDTree 模式
- [ ] 实现树形验证 (1 + tree_budget tokens)

---

### 优先级 5: 流式响应

#### 5.1 OpenAI 流式响应
- [ ] 实现 `handle_chat_completions_stream()` (当前是占位符)
- [ ] 使用 `text/event-stream` 格式
- [ ] 实现每个 token 的 SSE 事件
- [ ] 支持 `stream_options.include_usage`

**SSE 格式**:
```
data: {"id":"chatcmpl-xxx","choices":[{"index":0,"delta":{"content":"token"}}],"finish_reason":null}

data: [DONE]
```

#### 5.2 Anthropic 流式响应
- [ ] 完善 `handle_anthropic_messages_stream_impl()`
- [ ] 实现 Anthropic 事件格式:
  - `message_start`
  - `message_delta`
  - `message_stop`
  - `content_block_start`
  - `content_block_delta`
  - `content_block_stop`

---

### 优先级 6: Prefix Cache

#### 6.1 Cache 查找和分配
**已有组件**: `PrefixCache` 占位符 (line 65-104)

- [ ] 实现基于 prompt hash 的查找
- [ ] 实现 LRU 淘汰策略
- [ ] 支持 8 个 cache slot (可配置)

#### 6.2 Cache 快照和恢复
**已有组件**: `snapshot_target_cache()`, `restore_target_cache()`

- [ ] 在 prefill 后创建快照
- [ ] 在请求前查找匹配快照
- [ ] 实现快照恢复

---

### 优先级 7: 高级功能

#### 7.1 并发请求处理
- [ ] 实现请求队列
- [ ] 支持 GPU 并行 (多个 backend)
- [ ] 实现 batch 预处理

#### 7.2 采样参数
**已有组件**: `sampler.cpp` 提供完整支持

- [ ] 支持 `temperature`
- [ ] 支持 `top_p`
- [ ] 支持 `top_k`
- [ ] 支持 `repetition_penalty`
- [ ] 支持 `seed`

#### 7.3 工具调用集成
**已有组件**: `parsers.cpp` 已完成

- [x] 工具调用解析 ✅
- [ ] 在生成后调用 `parse_tool_calls()`
- [ ] 转换为 OpenAI tool_calls 格式
- [ ] 支持 OpenAI function calling

---

## 文件修改清单

### 需要修改的文件

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `src/server/dflash_server.cpp` | 移除占位符，集成真实实现 | P0 |
| `src/server/dflash_server.h` | 更新前向声明，添加成员变量 | P0 |
| `src/server/dflash_server.cpp` | 实现 `generate_internal()` | P1 |
| `src/server/dflash_server.cpp` | 实现模型加载 | P1 |
| `src/server/dflash_server.cpp` | 实现 Chat Template | P1 |

### 需要创建的文件

| 文件 | 内容 | 优先级 |
|------|------|--------|
| `src/server/chat_template.h` | Chat template 头文件 | P1 |
| `src/server/chat_template.cpp` | Chat template 实现 | P1 |
| `src/server/generation.h` | 生成循环辅助函数 | P1 |

### 可以复用的组件（无需修改）

| 组件 | 文件 | 说明 |
|------|------|------|
| Tokenizer | `src/server/tokenizer.cpp` | 直接使用 |
| 工具解析 | `src/server/parsers.cpp` | 直接使用 |
| 目标加载 | `src/gguf_target_loader.cpp` | 调用 API |
| 草稿加载 | `src/safetensors_draft.cpp` | 调用 API |
| 目标图 | `src/qwen35_target_graph.cpp` | 调用 API |
| 草稿图 | `src/qwen3_dflash_graph.cpp` | 调用 API |
| KV Cache | `src/kv_cache.cpp` | 调用 API |
| KV 量化 | `src/kv_quant.cpp` | 调用 API |
| PFlash | `src/flashprefill.cpp` | 调用 API |
| PFlash Drafter | `src/qwen3_drafter.cpp` | 调用 API |
| Sampler | `src/sampler.cpp` | 调用 API |

---

## 实现顺序建议

### 阶段 1: 基础功能 (1-2 天)
1. 移除占位符，集成真实 Tokenizer
2. 实现模型加载 (target + draft + cache)
3. 实现基础 generate_internal() (不含 DDTree/PFlash)

### 阶段 2: Chat Template (半天)
4. 实现 Chat Template 引擎
5. 测试 OpenAI Chat Completions API

### 阶段 3: DFlash 解码 (1 天)
6. 完整实现 DFlash 解码循环
7. 支持 draft + verify + accept + replay
8. 测试生成质量

### 阶段 4: PFlash 压缩 (1 天)
9. 集成 PFlash Drafter
10. 实现压缩协调逻辑
11. 测试长上下文性能

### 阶段 5: DDTree (1 天)
12. 提取 DDTree 代码
13. 实现树形验证
14. 测试接受率提升

### 阶段 6: TurboQuant (半天)
15. 集成 KV 类型解析
16. 支持环境变量配置
17. 测试内存和速度

### 阶段 7: 流式响应 (半天)
18. 实现 OpenAI SSE 流式
19. 实现 Anthropic SSE 流式
20. 测试流式延迟

### 阶段 8: 高级功能 (1-2 天)
21. 实现 Prefix Cache
22. 实现工具调用集成
23. 完善错误处理和日志

---

## API 设计

### 命令行参数

```bash
./dflash_server \
    --target <model.gguf> \
    --draft <draft.safetensors> \
    --host 0.0.0.0 \
    --port 8000 \
    --max-ctx 131072 \
    --kv-k-type q8_0 \
    --kv-v-type q8_0 \
    --prefill-mode auto \
    --prefill-threshold 8192 \
    --prefill-keep-ratio 0.1 \
    --tree-budget 22 \
    --fa-window 2048
```

### 环境变量

```bash
# KV Cache 类型
export DFLASH27B_KV_K=q8_0        # K 类型
export DFLASH27B_KV_V=q8_0        # V 类型
export DFLASH27B_KV_TQ3=1         # 启用 TurboQuant

# PFlash 调优
export DFLASH_FP_USE_BSA=1        # 启用 BSA kernel
export DFLASH_FP_ALPHA=0.12       # 选择阈值
```

---

## 参考

- `test/test_dflash.cpp` - 完整的 DFlash + DDTree 实现参考
- `src/internal.h` - 核心数据结构定义
- `scripts/server_tools.py` - Python 版本服务器逻辑
