"""
OpenAI-compatible HTTP server on top of test_dflash, **with tool-calling support**.

Patched fork of scripts/server.py that:
  1. Accepts the OpenAI `tools` array in ChatRequest.
  2. Renders tools into the prompt via Qwen's chat template (`tools=...`).
  3. Parses `<tool_call><function=...><parameter=...></tool_call>` blocks out
     of the model output and returns them as proper OpenAI `tool_calls`.
  4. Supports `role: "tool"` and assistant `tool_calls` in input messages so
     multi-turn agent loops round-trip correctly.

Streaming behavior:
  - Content tokens are streamed as `delta.content` until a `<tool_call>` opener
    is detected; the rest of the response is then buffered, parsed at the end
    of generation, and emitted as a single final `delta.tool_calls` chunk with
    `finish_reason: "tool_calls"`.
  - If no tool call appears in the output, behavior is identical to the
    upstream server.

Greedy decoding still applies (verify path is greedy-only). `temperature` and
`top_p` are accepted but ignored, matching upstream.

Run:
  pip install fastapi uvicorn transformers
  python3 scripts/server_tools.py --port 8000
"""
import argparse
import json
import logging
import os
import re
import struct
import subprocess
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware

from _prefill_hook import (
    PrefillConfig, add_cli_flags, config_from_args,
    compress_text_via_daemon,
)
from pydantic import BaseModel, Field
from starlette.concurrency import iterate_in_threadpool
from transformers import AutoTokenizer

from prefix_cache import DaemonStdoutBus, PrefixCache


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGET = Path(os.environ.get(
    "DFLASH_TARGET",
    str(ROOT / "models" / "Qwen3.6-27B-Q4_K_M.gguf"),
))
DEFAULT_DRAFT_ROOT = ROOT / "models" / "draft"
DEFAULT_BIN = ROOT / "build" / "test_dflash"
DEFAULT_PREFILL_DRAFTER = ROOT / "models" / "Qwen3-0.6B-BF16.gguf"
DEFAULT_BUDGET = 22
MODEL_NAME = "luce-dflash"


def resolve_draft(root: Path) -> Path:
    for st in root.rglob("model.safetensors"):
        return st
    raise FileNotFoundError(f"no model.safetensors under {root}")


# ─── pydantic schemas ──────────────────────────────────────────────

class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string per OpenAI spec


class ToolCall(BaseModel):
    id: str | None = None
    type: str = "function"
    function: ToolCallFunction


class ChatMessage(BaseModel):
    role: str
    content: Any | None = None  # str, list, or null when tool_calls present
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None


class ToolDef(BaseModel):
    type: str = "function"
    function: dict  # {name, description, parameters: {...JSON schema...}}


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[ChatMessage]
    stream: bool = False
    max_tokens: int = 50000
    temperature: float | None = None
    top_p: float | None = None
    tools: list[ToolDef] | None = None
    tool_choice: Any | None = None  # "auto" | "none" | {"function": {...}}
    chat_template_kwargs: dict | None = None  # e.g. {"enable_thinking": false}
    stop: Any | None = None  # str or list[str]
    stream_options: dict | None = None  # e.g. {"include_usage": true}


class AnthropicMessage(BaseModel):
    role: str
    # Anthropic allows either a plain string or a list of content blocks.
    content: str | list[dict]


class AnthropicMessagesRequest(BaseModel):
    model: str = MODEL_NAME
    max_tokens: int
    messages: list[AnthropicMessage]
    system: str | list[dict] | None = None
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    stop_sequences: list[str] | None = None


# ─── tool-call parser ──────────────────────────────────────────────

# Qwen3.6 chat template emits:
#   <tool_call>
#   <function=NAME>
#   <parameter=KEY>
#   VALUE
#   </parameter>
#   ...
#   </function>
#   </tool_call>
# Parsers ported from vLLM (Apache-2.0) for behavioral parity with
# `--reasoning-parser qwen3` and `--tool-call-parser qwen3_coder`:
#   vllm/reasoning/qwen3_reasoning_parser.py
#   vllm/tool_parsers/qwen3coder_tool_parser.py
# Core algorithms reproduced without vLLM runtime dependencies.

TOOL_CALL_COMPLETE_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
TOOL_CALL_FUNCTION_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL,
)
# vLLM's improved parameter regex: tolerates unclosed </parameter> by using
# next <parameter= or </function> or end-of-string as a terminator.
TOOL_CALL_PARAMETER_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)
TOOL_OPEN_TAG = "<tool_call>"

# Qwen3.6 chat template wraps the model's CoT inside <think>...</think>.
# The template typically prefills `<think>\n` into the prompt (headless mode)
# so only `</think>` appears in generated output; older templates emit both.
THINK_OPEN_TAG = "<think>"
THINK_CLOSE_TAG = "</think>"


def normalize_stop(stop) -> list[str]:
    """Coerce OpenAI's stop field (str | list[str] | None) to list[str]."""
    if not stop:
        return []
    if isinstance(stop, str):
        return [stop]
    return [s for s in stop if isinstance(s, str) and s]


def first_stop_match(text: str, stops: list[str]) -> int:
    """Return the earliest index where any stop sequence appears, or -1."""
    best = -1
    for s in stops:
        i = text.find(s)
        if i != -1 and (best == -1 or i < best):
            best = i
    return best


def parse_reasoning(text: str, thinking_enabled: bool = True) -> tuple[str, str | None]:
    """Port of vLLM's Qwen3ReasoningParser.extract_reasoning.

    Handles the three Qwen3.x thinking flavors:
      1. Paired:   `<think>...</think>` both in generated output.
      2. Headless: template prefilled `<think>\\n` into the prompt, model
         only emits `...</think>...`.
      3. Disabled: user passed `chat_template_kwargs: {enable_thinking: false}`.
         Template still emits `<think>\\n\\n</think>\\n\\n` but into the prompt;
         the model output is pure content and contains no tags.

    If the output was truncated mid-thinking (no `</think>` seen and
    `thinking_enabled=True`), returns `("", full_output_as_reasoning)` —
    matching vLLM's convention.

    Returns (cleaned_content, reasoning_content).
    """
    # Strip <think> if the model emitted it itself (older templates).
    parts = text.partition(THINK_OPEN_TAG)
    rest = parts[2] if parts[1] else parts[0]
    if THINK_CLOSE_TAG not in rest:
        if thinking_enabled:
            # No close tag — assume truncated; everything is reasoning.
            return "", (rest.strip() or None)
        else:
            # Thinking disabled — output is pure content.
            return rest.strip(), None
    reasoning, _, content = rest.partition(THINK_CLOSE_TAG)
    return content.strip(), (reasoning.strip() or None)


def _find_tool_properties(tools, function_name):
    """Helper matching vLLM's `find_tool_properties`: returns the parameters
    dict for a given function name, or {} if not found.
    Accepts pydantic ToolDef instances or plain dicts.
    """
    for t in tools or []:
        fn = t.function if hasattr(t, "function") else t.get("function", {})
        if hasattr(fn, "model_dump"):
            fn = fn.model_dump()
        if fn.get("name") == function_name:
            params = fn.get("parameters", {})
            if isinstance(params, dict):
                return params.get("properties", {})
    return {}


def _convert_param_value(param_value: str, param_name: str, param_config: dict,
                         func_name: str):
    """Port of vLLM's _convert_param_value. Coerces stringified XML values
    to their JSON-schema type (int/float/bool/object/array/string)."""
    import ast
    if param_value.lower() == "null":
        return None
    if param_name not in param_config:
        return param_value
    cfg = param_config[param_name]
    if isinstance(cfg, dict) and "type" in cfg:
        ptype = str(cfg["type"]).strip().lower()
    elif isinstance(cfg, dict) and "anyOf" in cfg:
        ptype = "object"
    else:
        ptype = "string"
    if ptype in ("string", "str", "text", "varchar", "char", "enum"):
        return param_value
    if any(ptype.startswith(p) for p in ("int", "uint", "long", "short", "unsigned")):
        try: return int(param_value)
        except (ValueError, TypeError): return param_value
    if ptype.startswith("num") or ptype.startswith("float"):
        try:
            f = float(param_value)
            return f if f - int(f) != 0 else int(f)
        except (ValueError, TypeError):
            return param_value
    if ptype in ("boolean", "bool", "binary"):
        return param_value.lower() == "true"
    # object / array / dict / list
    if (ptype in ("object", "array", "arr")
            or ptype.startswith("dict") or ptype.startswith("list")):
        try: return json.loads(param_value)
        except (json.JSONDecodeError, TypeError, ValueError): pass
    try: return ast.literal_eval(param_value)
    except (ValueError, SyntaxError, TypeError): return param_value


def parse_tool_calls(text: str, tools=None) -> tuple[str, list[dict]]:
    """Port of Qwen3CoderToolParser._parse_xml_function_call (non-streaming).

    Handles Qwen3.x's `<tool_call><function=NAME>...<parameter=KEY>VAL
    </parameter>...</function></tool_call>` XML. Uses vLLM's improved
    parameter regex that tolerates unclosed </parameter> tags. When `tools`
    is provided, each parameter value is coerced to its JSON-schema type.

    Returns (cleaned_content, tool_calls_list).
    """
    tool_calls: list[dict] = []
    cleaned_parts: list[str] = []
    cursor = 0
    for m in TOOL_CALL_COMPLETE_RE.finditer(text):
        cleaned_parts.append(text[cursor:m.start()])
        cursor = m.end()
        body = m.group(1)
        fn_match = TOOL_CALL_FUNCTION_RE.search(body)
        if not fn_match:
            continue
        fn_text = fn_match.group(1) or fn_match.group(2) or ""
        end_idx = fn_text.find(">")
        if end_idx == -1:
            continue
        function_name = fn_text[:end_idx].strip()
        params_region = fn_text[end_idx + 1:]
        param_config = _find_tool_properties(tools, function_name)
        args: dict = {}
        for match_text in TOOL_CALL_PARAMETER_RE.findall(params_region):
            eq_idx = match_text.find(">")
            if eq_idx == -1:
                continue
            k = match_text[:eq_idx].strip()
            v = match_text[eq_idx + 1:]
            if v.startswith("\n"): v = v[1:]
            if v.endswith("\n"): v = v[:-1]
            args[k] = _convert_param_value(v, k, param_config, function_name)
        tool_calls.append({
            "id": "call_" + uuid.uuid4().hex[:24],
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": json.dumps(args, ensure_ascii=False),
            },
        })
    cleaned_parts.append(text[cursor:])
    return "".join(cleaned_parts).strip(), tool_calls


# ─── app ───────────────────────────────────────────────────────────

def build_app(target: Path, draft: Path, bin_path: Path, budget: int,
              max_ctx: int, tokenizer: AutoTokenizer, stop_ids: set[int],
              prefill_cfg: PrefillConfig | None = None,
              drafter_tokenizer: AutoTokenizer | None = None,
              prefix_cache_slots: int = 4,
              prefill_cache_slots: int = 4,
              reasoning_reserve: int = 8000) -> FastAPI:
    import asyncio
    import logging
    app = FastAPI(title="Luce DFlash OpenAI server (tool-aware)")
    daemon_lock = asyncio.Lock()
    logger = logging.getLogger("server_tools")

    r_pipe, w_pipe = os.pipe()
    cmd = [str(bin_path), str(target), str(draft), "--daemon",
           "--fast-rollback", "--ddtree", f"--ddtree-budget={budget}",
           f"--max-ctx={max_ctx}",
           f"--stream-fd={w_pipe}"]
    daemon_proc = subprocess.Popen(cmd, pass_fds=(w_pipe,), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
    os.close(w_pipe)

    # 启动一个后台线程读取 daemon stderr 并记录
    import threading
    def _log_stderr():
        import sys
        while True:
            line = daemon_proc.stderr.readline()
            if not line:
                break
            line_str = line.decode('utf-8', errors='replace').rstrip('\n\r')
            print(f"[DAEMON STDERR] {line_str}", flush=True)
            logger.warning(f"[DAEMON STDERR] {line_str}")
    threading.Thread(target=_log_stderr, daemon=True).start()

    bus = DaemonStdoutBus(daemon_proc.stdout)
    # Mirror server.py: resolve effective KV-K type + FA window from env so
    # they participate in the prefix-cache hash key.
    def _resolve_kv_k_type():
        kv = "q8_0"
        if os.environ.get("DFLASH27B_KV_F16", "0") != "0":
            kv = "f16"
        if os.environ.get("DFLASH27B_KV_Q4", "0") != "0":
            kv = "q4_0"
        if os.environ.get("DFLASH27B_KV_TQ3", "0") != "0":
            kv = "tq3_0"
        if os.environ.get("DFLASH27B_KV_K"):
            kv = os.environ["DFLASH27B_KV_K"].lower()
        return kv
    _fa_window = int(os.environ.get("DFLASH27B_FA_WINDOW", 2048))
    prefix_cache = PrefixCache(
        daemon_stdin=daemon_proc.stdin,
        await_reply=bus.await_reply,
        daemon_lock=daemon_lock,
        tokenizer=tokenizer,
        kv_k_type=_resolve_kv_k_type(),
        fa_window=_fa_window,
        cap=prefix_cache_slots,
    )
    # Option 3: full-compress-result cache.  Only meaningful when pFlash
    # compression is enabled.  Uses a separate slot range [prefix_cap, ...).
    if prefill_cfg is not None and prefill_cache_slots > 0:
        prefix_cache.init_full_cache(prefill_cache_slots)

    @app.on_event("startup")
    async def _startup():
        import asyncio
        bus.start(asyncio.get_running_loop())
        await prefix_cache.startup_sync()

    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        import time
        import json
        start_time = time.time()
        path = request.url.path
        method = request.method

        # 记录请求基本信息
        logger.info(f"→ {method} {path} from {request.client.host if request.client else 'unknown'}")

        # 对于 POST 请求，尝试记录请求体（非流式）
        if method == "POST":
            try:
                body = await request.body()
                if body:
                    # 尝试解析 JSON，如果失败则记录原始内容
                    try:
                        body_json = json.loads(body.decode())
                        # 记录关键字段
                        msgs = body_json.get("messages", [])
                        logger.info(f"  Request: model={body_json.get('model')}, "
                                   f"messages={len(msgs)}, stream={body_json.get('stream')}, "
                                   f"max_tokens={body_json.get('max_tokens')}")
                        # 记录最后一条消息的长度
                        if msgs:
                            last_msg = msgs[-1]
                            content_len = len(last_msg.get("content", ""))
                            logger.info(f"  Last message: role={last_msg.get('role')}, content_length={content_len}")
                    except:
                        logger.info(f"  Request body: {body[:500]}")
            except Exception as e:
                logger.info(f"  (Failed to read body: {e})")

        try:
            response = await call_next(request)
            duration = time.time() - start_time
            # 记录响应状态
            log_msg = f"← {method} {path} - Status: {response.status_code} - {duration:.3f}s"
            if response.status_code >= 400:
                logger.warning(log_msg)
            else:
                logger.info(log_msg)
            return response
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"✗ {method} {path} - Error: {e} - {duration:.3f}s")
            raise

    # 全局异常处理器
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        import traceback
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": f"Internal server error: {str(exc)}"},
        )

    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        logger.error(f"Validation error on {request.method} {request.url.path}: {exc}")
        return JSONResponse(
            status_code=422,
            content={"detail": exc.errors()},
        )

    @app.get("/v1/models")
    def list_models():
        return {"object": "list",
                "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "luce"}]}

    def _maybe_compress_tool_chat(req: "ChatRequest", prompt_bin: Path,
                                  prompt_len: int, started_in_thinking: bool
                                  ) -> tuple[Path, int, bool]:
        """If prefill is on and the request has no tools and the last user
        message is long, run the daemon compress + re-tokenise. Returns
        (bin, prompt_len, started_in_thinking) — the last is recomputed when
        compression fires, otherwise passed through."""
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len, started_in_thinking
        if req.tools:
            # Tool definitions ride in the prompt; compressing them mangles JSON.
            return prompt_bin, prompt_len, started_in_thinking
        if not prefill_cfg.should_compress(prompt_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len, started_in_thinking

        last_user = next((m for m in reversed(req.messages) if m.role == "user"), None)
        if last_user is None or not isinstance(last_user.content, str):
            return prompt_bin, prompt_len, started_in_thinking

        # ── 额外检查：用户消息太短则不压缩 ────────────────────────────
        # 避免压缩太短的消息导致 drafter 返回空结果或垃圾数据
        user_msg_tokens = len(drafter_tokenizer.encode(last_user.content, add_special_tokens=False))
        MIN_USER_MSG_TOKENS = 100
        if user_msg_tokens < MIN_USER_MSG_TOKENS:
            return prompt_bin, prompt_len, started_in_thinking

        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=last_user.content,
            skip_park=prefill_cfg.skip_park,
        )

        # ── 诊断：检查压缩结果 ─────────────────────────────────
        if not compressed_text or not compressed_text.strip():
            logger.warning(f"[Prefill] compression returned empty! "
                          f"original_len={len(last_user.content)}, compressed_len={len(compressed_text)}")
            print(f"[PREFILL WARNING] compression returned empty, using original prompt")
            return prompt_bin, prompt_len, started_in_thinking

        new_msgs = []
        compressed_emitted = False
        for m in req.messages:
            if m is last_user and not compressed_emitted:
                new_msgs.append({"role": "user", "content": compressed_text})
                compressed_emitted = True
            else:
                d = {"role": m.role}
                if m.content is not None:
                    d["content"] = m.content
                new_msgs.append(d)

        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if req.chat_template_kwargs:
            kwargs.update(req.chat_template_kwargs)
        prompt = tokenizer.apply_chat_template(new_msgs, **kwargs)
        new_started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids), new_started_in_thinking

    def _tokenize_prompt(req: ChatRequest) -> tuple[Path, bool]:
        """Returns (prompt_bin_path, started_in_thinking). started_in_thinking
        is True when the chat template prefilled <think>\\n at the end of the
        prompt — the model's first emitted tokens are reasoning content."""
        # Convert pydantic messages to dicts the chat template expects.
        msgs: list[dict] = []
        for m in req.messages:
            d: dict = {"role": m.role}
            if m.content is not None:
                d["content"] = m.content
            if m.name is not None:
                d["name"] = m.name
            if m.tool_call_id is not None:
                d["tool_call_id"] = m.tool_call_id
            if m.tool_calls is not None:
                # The Qwen template walks tool_calls[i].function.{name, arguments}
                d["tool_calls"] = []
                for tc in m.tool_calls:
                    args = tc.function.arguments
                    # Template expects arguments as a dict, not a JSON string.
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args_obj = {"_raw": args}
                    else:
                        args_obj = args
                    d["tool_calls"].append({
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": args_obj},
                    })
            msgs.append(d)

        tools_arg = None
        if req.tools:
            tools_arg = [t.model_dump()["function"] | {"type": t.type} for t in req.tools]
            # The Qwen template accepts the raw OpenAI tools array structure.
            tools_arg = [t.model_dump() for t in req.tools]

        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools_arg:
            kwargs["tools"] = tools_arg
        # Per-request chat template knobs (e.g. enable_thinking, preserve_thinking).
        if req.chat_template_kwargs:
            kwargs.update(req.chat_template_kwargs)
        prompt = tokenizer.apply_chat_template(msgs, **kwargs)
        # Did the template prefill `<think>\n` at the end? Then streaming should
        # start in reasoning mode.
        started_in_thinking = bool(re.search(r"<think>\s*$", prompt))
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, started_in_thinking

    def _token_stream(r, n_gen):
        generated = 0
        hit_stop = False
        while True:
            b = os.read(r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    async def _drain_pipe_to_sentinel():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: list(_token_stream(r_pipe, 0)))

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        # 记录请求摘要
        msgs_count = len(req.messages) if req.messages else 0
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"[CHAT REQUEST] model={req.model or 'default'}, messages={msgs_count}, "
              f"max_tokens={req.max_tokens}, stream={req.stream}, tools={len(req.tools) if req.tools else 0}")
        # 记录每条消息的 role 和内容长度
        if req.messages:
            for i, m in enumerate(req.messages):
                content_len = len(m.content) if m.content else 0
                preview = m.content[:100] if m.content else ""
                print(f"  msg[{i}] role={m.role}, content_len={content_len}, preview={preview!r}")
        print(f"{separator}")
        logger.info(f"[Chat Request] model={req.model or 'default'}, messages={msgs_count}, "
                    f"max_tokens={req.max_tokens}, stream={req.stream}, tools={len(req.tools) if req.tools else 0}")

        prompt_bin, started_in_thinking = _tokenize_prompt(req)
        prompt_len = prompt_bin.stat().st_size // 4

        # Read back token ids for cache key (cheap — file is small).
        raw = prompt_bin.read_bytes()
        prompt_ids = [struct.unpack_from("<i", raw, i)[0]
                      for i in range(0, len(raw), 4)]

        completion_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())

        # --- Option 3: check full-compress-result cache before compression ---
        # This avoids the drafter dance entirely when we have a cached result.
        full_hit = None
        full_snap_prep = None
        cur_bin = prompt_bin  # default: no compression
        cur_ids = prompt_ids
        compression_fired = False

        async with daemon_lock:
            if prefill_cfg is not None and prefill_cfg.enabled and not req.tools:
                full_hit = prefix_cache.lookup_full(prompt_ids)

            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
                started_in_thinking = False  # cached result: no think prefill
            else:
                # pflash compress hook (no-op when off / has tools)
                # ── 必须在锁内同步执行压缩，避免竞态条件 ─────────────────
                new_bin, prompt_len, started_in_thinking = _maybe_compress_tool_chat(
                    req, prompt_bin, prompt_len, started_in_thinking)
                compression_fired = (new_bin != prompt_bin)
                cur_bin = new_bin
                if compression_fired:
                    # Read cur_ids from compressed bin for full-cache confirm.
                    raw_compressed = cur_bin.read_bytes()
                    cur_ids = [struct.unpack_from("<i", raw_compressed, i)[0]
                               for i in range(0, len(raw_compressed), 4)]
                    full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
                else:
                    cur_ids = prompt_ids

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)

        # ── 智能调整：thinking 模式下自动增加 gen_len ───────────────────
        # 检查是否启用 thinking 模式
        thinking_enabled = True
        if req.chat_template_kwargs:
            thinking_enabled = req.chat_template_kwargs.get("enable_thinking", True)

        if thinking_enabled and started_in_thinking:
            # 在 thinking 模式下，为推理预留额外 token
            # 如果用户设置的 max_tokens 太小，自动增加
            min_for_thinking = reasoning_reserve
            if gen_len < min_for_thinking:
                new_gen_len = min(min_for_thinking, available_gen)
                if new_gen_len > gen_len:
                    print(f"[AUTO-ADJUST] thinking mode: gen_len {gen_len} -> {new_gen_len} "
                          f"(reserved={reasoning_reserve}, available={available_gen})")
                    logger.info(f"[Auto-Adjust] thinking mode: gen_len {gen_len} -> {new_gen_len}, "
                                f"reserved={reasoning_reserve}")
                    gen_len = new_gen_len
        if gen_len <= 0:
            print(f"[400] Prompt too long: prompt_len={prompt_len}, max_ctx={max_ctx}, available_gen={available_gen}, req.max_tokens={req.max_tokens}")
            logger.warning(f"Prompt too long: prompt_len={prompt_len}, max_ctx={max_ctx}, available_gen={available_gen}, req.max_tokens={req.max_tokens}")
            if full_hit is None:
                try: cur_bin.unlink()
                except Exception: pass
            else:
                # On full-cache hit, cur_bin points at the persistent cached file
                # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                # the other hand, was never used (we hit before _maybe_compress) and
                # would otherwise leak.
                try: prompt_bin.unlink()
                except Exception: pass
            return JSONResponse(
                {"detail": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"},
                status_code=400)

        if req.stream:
            return await _stream_response(req, cur_bin, prompt_ids, gen_len,
                                           completion_id, created,
                                           started_in_thinking, daemon_lock,
                                           full_hit=full_hit,
                                           full_snap_prep=full_snap_prep,
                                           compression_fired=compression_fired,
                                           cur_ids=cur_ids)

        # Non-streaming: collect, parse, return.
        async with daemon_lock:
            print(f"[DEBUG] chat_completions: prompt_len={prompt_len}, gen_len={gen_len}, full_hit={full_hit is not None}")
            if full_hit is not None:
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                snap_prep = None
            elif compression_fired:
                if full_snap_prep is not None:
                    fslot, _ = full_snap_prep
                    cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                else:
                    cmd_line = f"{cur_bin} {gen_len}\n"
                snap_prep = None
            else:
                hit = prefix_cache.lookup(prompt_ids)
                snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                if hit:
                    slot, _prefix_len = hit
                    cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
                else:
                    cmd_line = f"{cur_bin} {gen_len}"
                if snap_prep:
                    cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                cmd_line += "\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = list(_token_stream(r_pipe, gen_len))
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, prompt_ids)
        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            # On full-cache hit, cur_bin points at the persistent cached file
            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
            # the other hand, was never used (we hit before _maybe_compress) and
            # would otherwise leak.
            try: prompt_bin.unlink()
            except Exception: pass

        print(f"[DEBUG] tokens from daemon: {len(tokens)} tokens")
        text = tokenizer.decode(tokens, skip_special_tokens=True)

        # ── 诊断：记录原始解码文本的特征 ──────────────────────────
        has_think_open = "<think>" in text
        has_think_close = "</think>" in text
        think_open_pos = text.find("<think>")
        think_close_pos = text.find("</think>")
        # 如果只有 <think> 没有 </think>，说明推理被截断
        truncated_reasoning = has_think_open and not has_think_close
        # 如果 </think> 在文本末尾附近，可能刚结束推理就耗尽了 token
        think_near_end = has_think_close and (len(text) - think_close_pos < 50)
        text_len = len(text)
        print(f"[DIAG] raw_text: len={text_len}, has_think_open={has_think_open}, "
              f"has_think_close={has_think_close}, truncated_reasoning={truncated_reasoning}, "
              f"think_near_end={think_near_end}")
        if has_think_close:
            after_think = text[think_close_pos + len("</think>"):]
            print(f"[DIAG] after </think>: len={len(after_think)}, content={after_think[:500]!r}")
        if truncated_reasoning:
            # 截断时记录推理文本最后部分
            print(f"[DIAG] TRUNCATED REASONING - last 200 chars of raw: {text[-200:]!r}")
        logger.info(f"[Diagnostic] raw_text_len={text_len}, has_think_open={has_think_open}, "
                    f"has_think_close={has_think_close}, truncated_reasoning={truncated_reasoning}, "
                    f"think_near_end={think_near_end}")

        # User-supplied stop sequences: trim at first match.
        stops = normalize_stop(req.stop)
        if stops:
            i = first_stop_match(text, stops)
            if i != -1:
                text = text[:i]
        # thinking_enabled was already set in the auto-adjust block above
        cleaned, tool_calls = parse_tool_calls(text, tools=req.tools)
        cleaned, reasoning = parse_reasoning(cleaned, thinking_enabled=thinking_enabled)

        # ── 诊断：parse_reasoning 结果 ──────────────────────────
        print(f"[DIAG] parse_reasoning result: reasoning_len={len(reasoning) if reasoning else 0}, "
              f"content_len={len(cleaned) if cleaned else 0}")
        if reasoning and not cleaned:
            # 有推理无内容时需要额外关注
            # 检查是否是模板没有添加 <think> 标签
            print(f"[DIAG] WARNING: reasoning present but NO content generated!")
            print(f"[DIAG] reasoning (last 500 chars): {reasoning[-500:]!r}")
            logger.warning(f"[Diagnostic] reasoning_only: reasoning_len={len(reasoning)}, "
                          f"content_empty=True, finish_reason will be 'reasoning_stop'")

        # Determine finish_reason
        finish_reason = "stop"
        if reasoning:
            finish_reason = "end_turn"

        # 记录完整响应文本
        separator = "=" * 60
        print(f"\n{separator}")
        print(f"[CHAT RESPONSE] prompt_len={prompt_len}, gen_tokens={len(tokens)}, "
              f"reasoning_len={len(reasoning) if reasoning else 0}, "
              f"content_len={len(cleaned) if cleaned else 0}, "
              f"tool_calls={len(tool_calls) if tool_calls else 0}")
        if reasoning:
            print(f"[CHAT RESPONSE] reasoning: {reasoning[:500]}")
        if cleaned:
            print(f"[CHAT RESPONSE] content: {cleaned[:500]}")
        if tool_calls:
            for tc in tool_calls:
                print(f"[CHAT RESPONSE] tool_call: {tc.get('function', {}).get('name', '?')} args={tc.get('function', {}).get('arguments', '')[:200]}")
        print(f"{separator}\n")

        logger.info(f"[Response] prompt_len={prompt_len}, gen_tokens={len(tokens)}, "
                    f"reasoning_len={len(reasoning) if reasoning else 0}, "
                    f"content_len={len(cleaned) if cleaned else 0}, "
                    f"tool_calls={len(tool_calls) if tool_calls else 0}, "
                    f"finish_reason={finish_reason if not reasoning else 'reasoning_stop'}")
        if reasoning:
            logger.info(f"[Response] reasoning_content (truncated): {reasoning[:500]}")
        if cleaned:
            logger.info(f"[Response] content (truncated): {cleaned[:500]}")
        if tool_calls:
            for tc in tool_calls:
                logger.info(f"[Response] tool_call: {tc.get('function', {}).get('name', '?')} args={tc.get('function', {}).get('arguments', '')[:200]}")

        # ── 诊断：无 content 时输出解决方案 ──────────────────────────
        if reasoning and not cleaned and not tool_calls:
            logger.warning(f"[DIAGNOSTIC] Model returned reasoning but NO content or tool_calls!")
            logger.warning(f"[DIAGNOSTIC] gen_tokens={len(tokens)}, reasoning_len={len(reasoning)}")
            logger.warning(f"[DIAGNOSTIC] If reasoning consumed all gen_tokens, increase max_tokens")
            logger.warning(f"[DIAGNOSTIC] To disable thinking: add chat_template_kwargs={{\"enable_thinking\": false}} to request")
            print(f"\n[DIAGNOSTIC] === No content returned, only reasoning ===")
            print(f"  gen_tokens: {len(tokens)}")
            print(f"  reasoning_len: {len(reasoning)} tokens")
            print(f"  content_len: 0")
            print(f"  SUGGESTIONS:")
            print(f"    1. Client-side: increase max_tokens (current={req.max_tokens})")
            print(f"    2. Client-side: disable thinking with chat_template_kwargs={{'enable_thinking': false}}")
            print(f"    3. Check if reasoning output is too long (last 200 chars: {reasoning[-200:]!r})")
            print(f"========================================\n")

        msg: dict = {"role": "assistant"}
        # finish_reason already initialized above, remove redundant assignment
        if reasoning:
            msg["reasoning_content"] = reasoning
        if tool_calls:
            msg["content"] = cleaned if cleaned else None
            msg["tool_calls"] = tool_calls
            finish_reason = "tool_calls"
        else:
            msg["content"] = cleaned

        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "message": msg,
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": prompt_len,
                      "completion_tokens": len(tokens),
                      "total_tokens": prompt_len + len(tokens)},
        })

    async def _stream_response(req, prompt_bin, prompt_ids, gen_len, completion_id,
                                created, started_in_thinking, lock,
                                full_hit=None, full_snap_prep=None,
                                compression_fired=False, cur_ids=None):
        # prompt_bin may be cur_bin (the compressed bin) when coming from the
        # compression or full-cache-hit path; prompt_len is derived from it.
        prompt_len = prompt_bin.stat().st_size // 4 if full_hit is None else (
            full_hit[2]  # cached_cur_ids_len
        )
        include_usage = bool(req.stream_options and req.stream_options.get("include_usage"))
        def chunk(delta_obj, finish=None):
            return {"id": completion_id, "object": "chat.completion.chunk",
                    "created": created, "model": MODEL_NAME,
                    "choices": [{"index": 0, "delta": delta_obj,
                                  "finish_reason": finish}]}

        async def sse() -> AsyncIterator[str]:
            print(f"[DEBUG] stream: prompt_len={prompt_len}, gen_len={gen_len}")
            async with lock:
                if full_hit is not None:
                    # Full-cache hit: RESTORE with cached cur_bin.
                    slot, cached_cur_bin, _cached_len = full_hit
                    cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                    snap_prep = None
                elif compression_fired:
                    # Compression fired on this request; set up full-cache snap.
                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        cmd_line = f"{prompt_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                    else:
                        cmd_line = f"{prompt_bin} {gen_len}\n"
                    snap_prep = None
                else:
                    hit = prefix_cache.lookup(prompt_ids)
                    snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                    if hit:
                        slot, _prefix_len = hit
                        cmd_line = f"RESTORE {slot} {prompt_bin} {gen_len}"
                    else:
                        cmd_line = f"{prompt_bin} {gen_len}"
                    if snap_prep:
                        cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                    cmd_line += "\n"
                daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                daemon_proc.stdin.flush()

                yield f"data: {json.dumps(chunk({'role': 'assistant'}))}\n\n"

                # State machine: mode ∈ {'reasoning', 'content', 'tool_buffer'}
                mode = "reasoning" if started_in_thinking else "content"
                window = ""           # holdback buffer for tag detection
                tool_buffer = ""
                stops = normalize_stop(req.stop)
                # Holdback must cover longest tag AND longest stop sequence.
                tag_holdback = max(len(THINK_OPEN_TAG), len(THINK_CLOSE_TAG), len(TOOL_OPEN_TAG))
                stop_holdback = max((len(s) for s in stops), default=0)
                HOLDBACK = max(tag_holdback, stop_holdback)
                completion_tokens = 0
                stop_hit = False

                def emit_delta(text, kind):
                    """kind: 'content' or 'reasoning_content'"""
                    if not text:
                        return None
                    return f"data: {json.dumps(chunk({kind: text}))}\n\n"

                try:
                    async for tok_id in iterate_in_threadpool(_token_stream(r_pipe, gen_len)):
                        completion_tokens += 1
                        piece = tokenizer.decode([tok_id])
                        window += piece

                        # Stop-sequence check on the visible (content/reasoning) stream.
                        if stops and mode != "tool_buffer":
                            si = first_stop_match(window, stops)
                            if si != -1:
                                window = window[:si]
                                stop_hit = True
                                # Flush truncated remainder per current mode.
                                kind = "reasoning_content" if mode == "reasoning" else "content"
                                out = emit_delta(window, kind)
                                if out: yield out
                                window = ""
                                break

                        # Process state transitions until no more tags found in window.
                        while True:
                            if mode == "tool_buffer":
                                tool_buffer += window
                                window = ""
                                break

                            # Look for the next tag of interest based on mode.
                            if mode == "reasoning":
                                idx = window.find(THINK_CLOSE_TAG)
                                if idx != -1:
                                    pre = window[:idx]
                                    out = emit_delta(pre, "reasoning_content")
                                    if out: yield out
                                    window = window[idx + len(THINK_CLOSE_TAG):]
                                    mode = "content"
                                    continue
                                # No close tag yet. Stream all but holdback.
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "reasoning_content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                            else:  # mode == "content"
                                think_idx = window.find(THINK_OPEN_TAG)
                                tool_idx  = window.find(TOOL_OPEN_TAG)
                                # Pick the earliest tag that actually appears.
                                hits = [(i, t) for i, t in
                                        ((think_idx, "think"), (tool_idx, "tool")) if i != -1]
                                if hits:
                                    hits.sort()
                                    idx, which = hits[0]
                                    pre = window[:idx]
                                    out = emit_delta(pre, "content")
                                    if out: yield out
                                    if which == "think":
                                        window = window[idx + len(THINK_OPEN_TAG):]
                                        mode = "reasoning"
                                    else:  # tool
                                        tool_buffer = window[idx:]
                                        window = ""
                                        mode = "tool_buffer"
                                    continue
                                if len(window) > HOLDBACK:
                                    safe = window[:-HOLDBACK]
                                    out = emit_delta(safe, "content")
                                    if out: yield out
                                    window = window[-HOLDBACK:]
                                break  # need more tokens

                    if stop_hit:
                        finish_reason = "stop"
                        yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                        if include_usage:
                            usage_chunk = {"id": completion_id, "object": "chat.completion.chunk",
                                           "created": created, "model": MODEL_NAME, "choices": [],
                                           "usage": {"prompt_tokens": prompt_len,
                                                      "completion_tokens": completion_tokens,
                                                      "total_tokens": prompt_len + completion_tokens}}
                            yield f"data: {json.dumps(usage_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        if full_hit is None:
                            try: prompt_bin.unlink()
                            except Exception: pass
                        return

                    # Generation done. Flush remaining window per current mode.
                    if mode == "reasoning" and window:
                        out = emit_delta(window, "reasoning_content")
                        if out: yield out
                    elif mode == "content" and window:
                        out = emit_delta(window, "content")
                        if out: yield out
                    elif mode == "tool_buffer":
                        tool_buffer += window
                    window = ""

                    finish_reason = "stop"
                    if mode == "tool_buffer":
                        cleaned_after, tool_calls = parse_tool_calls(tool_buffer, tools=req.tools)
                        if tool_calls:
                            if cleaned_after:
                                out = emit_delta(cleaned_after, "content")
                                if out: yield out
                            tc_delta_list = [{
                                "index": i, "id": tc["id"], "type": "function",
                                "function": {"name": tc["function"]["name"],
                                              "arguments": tc["function"]["arguments"]},
                            } for i, tc in enumerate(tool_calls)]
                            yield f"data: {json.dumps(chunk({'tool_calls': tc_delta_list}))}\n\n"
                            finish_reason = "tool_calls"
                        else:
                            # Unclosed <tool_call> — emit raw as content fallback.
                            out = emit_delta(tool_buffer, "content")
                            if out: yield out
                finally:
                    if full_hit is None:
                        try: prompt_bin.unlink()
                        except Exception: pass

                if full_snap_prep is not None:
                    fslot, _ = full_snap_prep
                    prefix_cache.confirm_full_snap(fslot, prompt_ids, prompt_bin, len(cur_ids))
                elif snap_prep:
                    prefix_cache.confirm_inline_snap(*snap_prep, prompt_ids)

                # 记录流式响应完成
                separator = "=" * 60
                print(f"\n{separator}")
                print(f"[STREAM RESPONSE] prompt_len={prompt_len}, completion_tokens={completion_tokens}, "
                      f"finish_reason={finish_reason}, mode={mode if mode else 'unknown'}")
                # ── 诊断：流式结束在 reasoning 模式 ──────────────────────────
                if mode == "reasoning" and not window:
                    # 流式结束时还在 reasoning 模式，且没有剩余内容
                    print(f"[DIAGNOSTIC] Stream ended in reasoning mode with no content!")
                    print(f"  completion_tokens: {completion_tokens}")
                    print(f"  SUGGESTIONS:")
                    print(f"    1. Increase max_tokens (current={req.max_tokens})")
                    print(f"    2. Disable thinking: chat_template_kwargs={{'enable_thinking': false}}")
                    logger.warning(f"[Stream Diagnostic] Ended in reasoning mode, tokens={completion_tokens}")
                print(f"{separator}\n")
                logger.info(f"[Stream Response] prompt_len={prompt_len}, completion_tokens={completion_tokens}, "
                            f"finish_reason={finish_reason}, mode={mode if mode else 'unknown'}")

                yield f"data: {json.dumps(chunk({}, finish=finish_reason))}\n\n"
                if include_usage:
                    usage_chunk = {
                        "id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": MODEL_NAME,
                        "choices": [],
                        "usage": {"prompt_tokens": prompt_len,
                                   "completion_tokens": completion_tokens,
                                   "total_tokens": prompt_len + completion_tokens},
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    # ── Anthropic Messages API ──────────────────────────────────────
    # Mirrors the OpenAI endpoint but formatted for the Anthropic SDK
    # (Claude Code, Anthropic clients). Tool calling NOT forwarded here
    # yet — agent CLIs that want tools should use /v1/chat/completions.

    def _anthropic_text_from_content(content) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "".join(parts)

    def _tokenize_anthropic(req: AnthropicMessagesRequest
                            ) -> tuple[Path, int, list[dict]]:
        msgs = []
        system_text = _anthropic_text_from_content(req.system) if req.system else None
        if system_text:
            msgs.append({"role": "system", "content": system_text})
        for m in req.messages:
            msgs.append({"role": m.role,
                         "content": _anthropic_text_from_content(m.content)})
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        # mkstemp returns (fd, path); discarding fd leaks 1 per request (#15).
        fd, path = tempfile.mkstemp(suffix=".bin")
        tmp = Path(path)
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        return tmp, len(ids), msgs

    def _maybe_compress_anthropic(prompt_bin: Path, prompt_len: int,
                                  msgs: list[dict]) -> tuple[Path, int]:
        if not prefill_cfg or not prefill_cfg.enabled:
            return prompt_bin, prompt_len
        if not prefill_cfg.should_compress(prompt_len) or drafter_tokenizer is None:
            return prompt_bin, prompt_len
        last_user_idx = next((i for i in range(len(msgs) - 1, -1, -1)
                              if msgs[i]["role"] == "user"), None)
        if last_user_idx is None:
            return prompt_bin, prompt_len
        long_text = msgs[last_user_idx]["content"]
        compressed_text = compress_text_via_daemon(
            daemon_stdin=daemon_proc.stdin,
            r_pipe=r_pipe,
            drafter_tokenizer=drafter_tokenizer,
            cfg=prefill_cfg,
            prompt_text=long_text,
            skip_park=prefill_cfg.skip_park,
        )
        new_msgs = list(msgs)
        new_msgs[last_user_idx] = {"role": "user", "content": compressed_text}
        prompt = tokenizer.apply_chat_template(
            new_msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in ids:
                f.write(struct.pack("<i", int(t)))
        try: prompt_bin.unlink()
        except Exception: pass
        return Path(path), len(ids)

    async def _astream_tokens(r, n_gen):
        """Yields one token at a time without blocking the event loop.
        Each 4-byte pipe read is dispatched to a worker thread."""
        generated = 0
        hit_stop = False
        while True:
            b = await asyncio.to_thread(os.read, r, 4)
            if not b or len(b) < 4:
                break
            tok_id = struct.unpack("<i", b)[0]
            if tok_id == -1:
                break
            if hit_stop:
                continue
            if tok_id in stop_ids:
                hit_stop = True
                continue
            generated += 1
            yield tok_id
            if generated >= n_gen:
                hit_stop = True

    @app.post("/v1/messages")
    async def anthropic_messages(req: AnthropicMessagesRequest):
        # 记录请求摘要
        msgs_count = len(req.messages) if req.messages else 0
        last_msg = req.messages[-1] if req.messages else None
        last_content = ""
        if last_msg:
            if isinstance(last_msg.content, str):
                last_content = last_msg.content[:200]
            elif isinstance(last_msg.content, list):
                last_content = str(last_msg.content[:2])
        logger.info(f"[Anthropic Request] model={req.model}, messages={msgs_count}, "
                    f"max_tokens={req.max_tokens}, stream={req.stream}")
        if last_msg:
            logger.info(f"[Anthropic Request] last_msg: role={last_msg.role}, content={last_content!r}")

        prompt_bin, prompt_len, raw_msgs = _tokenize_anthropic(req)

        # Read raw prompt_ids BEFORE compression (for full-cache key).
        raw = prompt_bin.read_bytes()
        prompt_ids = [struct.unpack_from("<i", raw, i)[0]
                      for i in range(0, len(raw), 4)]

        full_hit = None
        full_snap_prep = None
        cur_bin = prompt_bin
        cur_ids = prompt_ids
        compression_fired = False

        async with daemon_lock:
            full_hit = prefix_cache.lookup_full(prompt_ids)
            if full_hit is not None:
                slot, cached_cur_bin, cached_cur_ids_len = full_hit
                cur_bin = Path(cached_cur_bin)
                cur_ids = None
                prompt_len = cached_cur_ids_len
            else:
                # ── 必须在锁内同步执行压缩，避免竞态条件 ─────────────────
                new_bin, new_len = _maybe_compress_anthropic(prompt_bin, prompt_len, raw_msgs)
                compression_fired = (new_bin != prompt_bin)
                cur_bin = new_bin
                prompt_len = new_len
                if compression_fired:
                    raw_compressed = cur_bin.read_bytes()
                    cur_ids = [struct.unpack_from("<i", raw_compressed, i)[0]
                               for i in range(0, len(raw_compressed), 4)]
                    full_snap_prep = prefix_cache.prepare_full_snap(prompt_ids)
                else:
                    cur_ids = prompt_ids

        available_gen = max_ctx - prompt_len - 20
        gen_len = min(req.max_tokens, available_gen)
        if gen_len <= 0:
            print(f"[400] Prompt too long: prompt_len={prompt_len}, max_ctx={max_ctx}, available_gen={available_gen}, req.max_tokens={req.max_tokens}")
            logger.warning(f"Prompt too long: prompt_len={prompt_len}, max_ctx={max_ctx}, available_gen={available_gen}, req.max_tokens={req.max_tokens}")
            if full_hit is None:
                try: cur_bin.unlink()
                except Exception: pass
            else:
                # On full-cache hit, cur_bin points at the persistent cached file
                # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                # the other hand, was never used (we hit before _maybe_compress) and
                # would otherwise leak.
                try: prompt_bin.unlink()
                except Exception: pass
            return JSONResponse(
                {"type": "error",
                 "error": {"type": "invalid_request_error",
                           "message": f"Prompt length ({prompt_len}) exceeds max_ctx ({max_ctx})"}},
                status_code=400)

        msg_id = "msg_" + uuid.uuid4().hex[:24]

        if req.stream:
            async def sse() -> AsyncIterator[str]:
                async with daemon_lock:
                    if full_hit is not None:
                        slot, cached_cur_bin, _cached_len = full_hit
                        cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                        snap_prep = None
                    elif compression_fired:
                        if full_snap_prep is not None:
                            fslot, _ = full_snap_prep
                            cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                        else:
                            cmd_line = f"{cur_bin} {gen_len}\n"
                        snap_prep = None
                    else:
                        hit = prefix_cache.lookup(prompt_ids)
                        snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                        if hit:
                            slot, _prefix_len = hit
                            cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
                        else:
                            cmd_line = f"{cur_bin} {gen_len}"
                        if snap_prep:
                            cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                        cmd_line += "\n"

                    message_start = {
                        "type": "message_start",
                        "message": {
                            "id": msg_id, "type": "message", "role": "assistant",
                            "model": req.model or MODEL_NAME,
                            "content": [], "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": prompt_len, "output_tokens": 0},
                        },
                    }
                    yield f"event: message_start\ndata: {json.dumps(message_start)}\n\n"

                    cb_start = {
                        "type": "content_block_start", "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    }
                    yield f"event: content_block_start\ndata: {json.dumps(cb_start)}\n\n"

                    daemon_proc.stdin.write(cmd_line.encode("utf-8"))
                    daemon_proc.stdin.flush()

                    out_tokens = 0
                    try:
                        async for tok_id in _astream_tokens(r_pipe, gen_len):
                            out_tokens += 1
                            delta = {
                                "type": "content_block_delta", "index": 0,
                                "delta": {"type": "text_delta",
                                          "text": tokenizer.decode([tok_id])},
                            }
                            yield f"event: content_block_delta\ndata: {json.dumps(delta)}\n\n"
                    finally:
                        if full_hit is None:
                            try: cur_bin.unlink()
                            except Exception: pass
                        else:
                            # On full-cache hit, cur_bin points at the persistent cached file
                            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
                            # the other hand, was never used (we hit before _maybe_compress) and
                            # would otherwise leak.
                            try: prompt_bin.unlink()
                            except Exception: pass

                    if full_snap_prep is not None:
                        fslot, _ = full_snap_prep
                        prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
                    elif snap_prep:
                        prefix_cache.confirm_inline_snap(*snap_prep, prompt_ids)

                    yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

                    msg_delta = {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                        "usage": {"output_tokens": out_tokens},
                    }
                    yield f"event: message_delta\ndata: {json.dumps(msg_delta)}\n\n"
                    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        # Non-streaming
        async with daemon_lock:
            if full_hit is not None:
                slot, cached_cur_bin, _cached_len = full_hit
                cmd_line = f"RESTORE {slot} {cached_cur_bin} {gen_len}\n"
                snap_prep = None
            elif compression_fired:
                if full_snap_prep is not None:
                    fslot, _ = full_snap_prep
                    cmd_line = f"{cur_bin} {gen_len} snap={len(cur_ids)}:{fslot}\n"
                else:
                    cmd_line = f"{cur_bin} {gen_len}\n"
                snap_prep = None
            else:
                hit = prefix_cache.lookup(prompt_ids)
                snap_prep = prefix_cache.prepare_inline_snap(prompt_ids)
                if hit:
                    slot, _prefix_len = hit
                    cmd_line = f"RESTORE {slot} {cur_bin} {gen_len}"
                else:
                    cmd_line = f"{cur_bin} {gen_len}"
                if snap_prep:
                    cmd_line += f" snap={snap_prep[1]}:{snap_prep[0]}"
                cmd_line += "\n"
            daemon_proc.stdin.write(cmd_line.encode("utf-8"))
            daemon_proc.stdin.flush()
            tokens = [t async for t in _astream_tokens(r_pipe, gen_len)]
            if full_snap_prep is not None:
                fslot, _ = full_snap_prep
                prefix_cache.confirm_full_snap(fslot, prompt_ids, cur_bin, len(cur_ids))
            elif snap_prep:
                prefix_cache.confirm_inline_snap(*snap_prep, prompt_ids)

        if full_hit is None:
            try: cur_bin.unlink()
            except Exception: pass
        else:
            # On full-cache hit, cur_bin points at the persistent cached file
            # (which we MUST keep). The tokenize-stage prompt_bin tempfile, on
            # the other hand, was never used (we hit before _maybe_compress) and
            # would otherwise leak.
            try: prompt_bin.unlink()
            except Exception: pass
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        return JSONResponse({
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": req.model or MODEL_NAME,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": prompt_len,
                      "output_tokens": len(tokens)},
        })

    return app


def setup_logging(log_file: str | None = None):
    """配置日志系统，输出到文件和终端"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    handlers = []

    # 文件日志
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(log_format, date_format))
        handlers.append(file_handler)

    # 终端日志
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    handlers.append(console_handler)

    # 配置根日志记录器
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        datefmt=date_format,
        handlers=handlers
    )
    return logging.getLogger("server_tools")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    ap.add_argument("--draft",  type=Path, default=DEFAULT_DRAFT_ROOT)
    ap.add_argument("--bin",    type=Path, default=DEFAULT_BIN)
    ap.add_argument("--budget", type=int,  default=DEFAULT_BUDGET)
    ap.add_argument("--log-file", type=str, default=None,
                    help="日志文件路径 (默认: logs/server_YYYYMMDD_HHMMSS.log)")
    # Attention compute currently scales with --max-ctx, not the actual
    # prompt+gen length (see issue #10). Default 16384 fits most API
    # workloads without the 20×+ slowdown users hit with --max-ctx=131072
    # on short requests. Bump via --max-ctx for long-context serving.
    ap.add_argument("--max-ctx", type=int, default=131072,
                    help="Maximum context length (default: 131072; oversizing "
                         "this, e.g. 131072 on short prompts, can slow "
                         "attention 20×+ until issue #10 is fixed)")
    ap.add_argument("--kv-f16", action="store_true",
                    help="Force F16 KV cache. When --max-ctx > 48K the server "
                         "auto-enables TQ3_0 KV to fit; pass --kv-f16 to opt out.")
    ap.add_argument("--cache-type-k", "--ctk", dest="cache_type_k", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="K cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for K). "
                         "See kv_quant.cpp for supported (K,V) pairs.")
    ap.add_argument("--cache-type-v", "--ctv", dest="cache_type_v", default=None,
                    choices=["f16","bf16","q4_0","q4_1","q5_0","q5_1","q8_0","tq3_0"],
                    help="V cache element type (overrides --kv-q4/--kv-tq3/--kv-f16 for V).")
    ap.add_argument("--fa-window", type=int, default=None,
                    help="Sliding window for FA layers (KV positions). 0 = full "
                         "attention. Default 2048 (set in C++); only kicks in "
                         "once kv_cache > window. Trades attention range for "
                         "long-context decode speed.")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-27B",
                    help="HF tokenizer id; Qwen3.6 shares this tokenizer.")
    add_cli_flags(ap)
    ap.add_argument("--prefix-cache-slots", type=int, default=4,
                    help="Number of prefix-cache snapshot slots (0 to disable)")
    ap.add_argument("--prefill-cache-slots", type=int, default=4,
                    help="Number of full-compress-result cache slots (Option 3). "
                         "Only active when --prefill-compression is enabled. "
                         "prefix-cache-slots + prefill-cache-slots must not exceed 8.")
    ap.add_argument("--reasoning-reserve", type=int, default=8000,
                    help="Extra tokens to reserve when thinking is enabled. "
                    "If enable_thinking=true and max_tokens < reasoning_reserve, "
                    "server will auto-increase max_tokens by this amount. "
                    "Default: 8000 (prevents reasoning-only responses).")
    args = ap.parse_args()
    prefill_cfg = config_from_args(args, DEFAULT_PREFILL_DRAFTER)

    # Auto-enable TQ3_0 KV cache when the requested context exceeds what F16 fits.
    # setdefault so an explicit user DFLASH27B_KV_TQ3=0 still wins.
    if args.cache_type_k:
        os.environ["DFLASH27B_KV_K"] = args.cache_type_k
    if args.cache_type_v:
        os.environ["DFLASH27B_KV_V"] = args.cache_type_v
    # TQ3_0 threshold: enable when max_ctx > 48K (F16 fits ~48K with current memory)
    if args.max_ctx > 49152 and not args.kv_f16 and not args.cache_type_k and not args.cache_type_v:
        os.environ.setdefault("DFLASH27B_KV_TQ3", "1")

    if args.fa_window is not None:
        os.environ["DFLASH27B_FA_WINDOW"] = str(args.fa_window)

    # When pflash is on, daemon needs the same env the bench harness uses
    # (otherwise post-compress draft graph reserve OOMs at 64K+).
    if args.prefill_compression != "off":
        os.environ.setdefault("DFLASH27B_LM_HEAD_FIX", "0")
        os.environ.setdefault("DFLASH27B_FA_WINDOW", "0")
        # FlashPrefill bench-tuned defaults from pflash/README.md headline numbers
        # (10x TTFT @ 64K). Without these the drafter falls through to the WMMA
        # fallback at the default ALPHA=0.12, which roughly triples cold-start
        # TTFT. setdefault so explicit user overrides still win.
        os.environ.setdefault("DFLASH_FP_USE_BSA", "1")
        os.environ.setdefault("DFLASH_FP_ALPHA",   "0.85")

    if not args.bin.is_file():
        raise SystemExit(f"binary not found at {args.bin}")
    if not args.target.is_file():
        raise SystemExit(f"target GGUF not found at {args.target}")
    draft = resolve_draft(args.draft) if args.draft.is_dir() else args.draft
    if not draft.is_file():
        raise SystemExit(f"draft safetensors not found at {args.draft}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids: stop_ids.add(ids[0])

    drafter_tokenizer = None
    if prefill_cfg.enabled:
        drafter_tokenizer = AutoTokenizer.from_pretrained(
            prefill_cfg.drafter_tokenizer_id, trust_remote_code=True)

    # 配置日志文件
    if args.log_file is None:
        # 默认日志文件路径
        logs_dir = ROOT / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        args.log_file = str(logs_dir / f"server_{timestamp}.log")

    logger = setup_logging(args.log_file)
    logger.info("=== Luce DFlash Server 启动 ===")
    logger.info(f"日志文件: {args.log_file}")
    logger.info(f"target = {args.target}")
    logger.info(f"draft  = {draft}")
    logger.info(f"bin    = {args.bin}")
    logger.info(f"budget = {args.budget}")
    logger.info(f"max_ctx= {args.max_ctx}")
    logger.info(f"tokenizer = {args.tokenizer}")
    if prefill_cfg.enabled:
        logger.info(f"pflash = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
                    f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        logger.info("pflash = off")

    import uvicorn
    print(f"Luce DFlash OpenAI server (tool-aware) on http://{args.host}:{args.port}")
    print(f"  target = {args.target}")
    print(f"  draft  = {draft}")
    print(f"  bin    = {args.bin}")
    print(f"  budget = {args.budget}")
    print(f"  max_ctx= {args.max_ctx}")
    print(f"  tokenizer = {args.tokenizer}")
    if prefill_cfg.enabled:
        print(f"  pflash = {prefill_cfg.mode} · threshold={prefill_cfg.threshold} "
              f"keep={prefill_cfg.keep_ratio} drafter={prefill_cfg.drafter_gguf}")
    else:
        print("  pflash = off")
    print(f"  日志文件: {args.log_file}")
    app = build_app(bin_path=args.bin,
                     target=args.target,
                     draft=draft,
                     budget=args.budget,
                     max_ctx=args.max_ctx,
                     tokenizer=tokenizer,
                     stop_ids=stop_ids,
                     prefill_cfg=prefill_cfg,
                     drafter_tokenizer=drafter_tokenizer,
                     prefix_cache_slots=args.prefix_cache_slots,
                     prefill_cache_slots=args.prefill_cache_slots,
                     reasoning_reserve=args.reasoning_reserve)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
