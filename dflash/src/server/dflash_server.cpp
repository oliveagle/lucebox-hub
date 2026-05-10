// dflash_server.cpp - C++ HTTP Server Implementation
//
// This file implements a standalone HTTP server for DFlash that replaces
// the Python-based server_tools.py.

#include "dflash_server.h"
#include "internal.h"
#include "parsers.h"
#include "tokenizer.h"
#include "kv_quant.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <optional>

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace dflash27b {

// Use the real Tokenizer implementation from tokenizer.h
// It's in the dflash27b namespace, so we can use it directly

// ─── Prefix Cache ───────────────────────────────────────────────────────

class PrefixCache {
public:
    static constexpr int MAX_SLOTS = 8;

    struct Slot {
        bool occupied = false;
        std::vector<int32_t> prompt_ids;
        // TODO: Add GGML buffer pointers
    };

    std::optional<int> find_slot(const std::vector<int32_t>& prompt_ids) {
        // Simple linear search for matching prompt
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (slots_[i].occupied && slots_[i].prompt_ids == prompt_ids) {
                return i;
            }
        }
        return std::nullopt;
    }

    int allocate_slot() {
        for (int i = 0; i < MAX_SLOTS; i++) {
            if (!slots_[i].occupied) {
                slots_[i].occupied = true;
                return i;
            }
        }
        return -1;  // Full
    }

    void free_slot(int slot) {
        if (slot >= 0 && slot < MAX_SLOTS) {
            slots_[slot].occupied = false;
            slots_[slot].prompt_ids.clear();
        }
    }

private:
    std::array<Slot, MAX_SLOTS> slots_;
};

// ─── Prefill Coordinator ────────────────────────────────────────────────

class PrefillCoordinator {
public:
    bool should_compress(int prompt_len, bool has_tools) {
        if (!config_.enabled) return false;
        if (has_tools) return false;
        if (config_.mode == "always") return true;
        if (config_.mode == "auto") return prompt_len >= config_.threshold;
        return false;
    }

    std::vector<int32_t> compress(const std::vector<int32_t>& tokens) {
        // TODO: Implement PFlash compression
        return tokens;
    }

    struct Config {
        bool enabled = true;
        std::string mode = "always";
        int threshold = 8192;
        float keep_ratio = 0.1f;
        std::string drafter_path;
    } config_;
};

// ─── DFlashServer Implementation ────────────────────────────────────────

DFlashServer::DFlashServer() = default;

DFlashServer::~DFlashServer() {
    stop();
}

bool DFlashServer::initialize(const ServerConfig& config) {
    config_ = config;

    // Create HTTP server
    http_server_ = std::make_unique<httplib::Server>();
    http_server_->set_logger([](const httplib::Request& req, const httplib::Response& res) {
        std::cout << "[" << req.method << "] " << req.path << " -> " << res.status << std::endl;
    });

    // Initialize tokenizer
    tokenizer_ = std::make_unique<Tokenizer>();
    if (!tokenizer_->load_from_gguf(config.target_path)) {
        std::cerr << "Failed to load tokenizer: " << tokenizer_->last_error() << std::endl;
        return false;
    }

    // Initialize prefix cache
    prefix_cache_ = std::make_unique<PrefixCache>();

    // Initialize prefill coordinator
    prefill_ = std::make_unique<PrefillCoordinator>();
    prefill_->config_ = {
        .enabled = config.prefill_enabled,
        .mode = config.prefill_mode,
        .threshold = config.prefill_threshold,
        .keep_ratio = config.prefill_keep_ratio,
        .drafter_path = config.prefill_drafter
    };

    // Resolve KV cache types from environment variables
    ggml_type kv_k_type, kv_v_type;
    dflash::resolve_kv_types(kv_k_type, kv_v_type);
    std::cout << "KV cache types: K=" << dflash::kv_type_name(kv_k_type)
              << ", V=" << dflash::kv_type_name(kv_v_type) << std::endl;

    // Create GPU backend for model loading
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (!backend) {
        std::cerr << "Failed to create CUDA backend" << std::endl;
        return false;
    }
    std::cout << "Created CUDA backend" << std::endl;

    // Load target model (GGUF)
    target_weights_ = std::make_unique<TargetWeights>();
    if (!load_target_gguf(config.target_path, backend, *target_weights_)) {
        std::cerr << "Failed to load target model from: " << config.target_path << std::endl;
        ggml_backend_free(backend);
        return false;
    }
    std::cout << "Loaded target model: " << target_weights_->n_embd << " dims, "
              << target_weights_->n_layer << " layers" << std::endl;

    // Load draft model (safetensors or GGUF)
    draft_weights_ = std::make_unique<DraftWeights>();
    bool is_gguf = config.draft_path.size() >= 5 &&
                   config.draft_path.substr(config.draft_path.size() - 5) == ".gguf";

    if (is_gguf) {
        if (!load_draft_gguf(config.draft_path, backend, *draft_weights_)) {
            std::cerr << "Failed to load draft model from: " << config.draft_path << std::endl;
            ggml_backend_free(backend);
            return false;
        }
    } else {
        if (!load_draft_safetensors(config.draft_path, backend, *draft_weights_)) {
            std::cerr << "Failed to load draft model from: " << config.draft_path << std::endl;
            ggml_backend_free(backend);
            return false;
        }
    }
    std::cout << "Loaded draft model: " << draft_weights_->n_embd << " dims, "
              << draft_weights_->n_layer << " layers" << std::endl;

    // Create target KV cache
    cache_ = std::make_unique<TargetCache>();
    int max_verify_tokens = config.ddtree_budget + 1;  // For DDTree
    if (!create_target_cache(*target_weights_, config.max_ctx, max_verify_tokens,
                             backend, *cache_)) {
        std::cerr << "Failed to create target cache" << std::endl;
        ggml_backend_free(backend);
        return false;
    }
    std::cout << "Created target cache: max_ctx=" << cache_->max_ctx << std::endl;

    setup_routes();

    return true;
}

void DFlashServer::setup_routes() {
    // List models
    http_server_->Get("/v1/models",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_models(req, res);
        }
    );

    // Chat completions (non-streaming)
    http_server_->Post("/v1/chat/completions",
        [this](const httplib::Request& req, httplib::Response& res) {
            if (is_streaming_request(req)) {
                handle_chat_completions_stream(req, res);
            } else {
                handle_chat_completions(req, res);
            }
        }
    );

    // Health check
    http_server_->Get("/health",
        [](const httplib::Request& req, httplib::Response& res) {
            json health = {{"status", "ok"}, {"model", "dflash"}};
            res.set_content(health.dump(), "application/json");
        }
    );

    // Anthropic Messages API
    http_server_->Post("/v1/messages",
        [this](const httplib::Request& req, httplib::Response& res) {
            handle_anthropic_messages(req, res);
        }
    );
}

bool DFlashServer::is_streaming_request(const httplib::Request& req) {
    try {
        json body = json::parse(req.body);
        return body.value("stream", false);
    } catch (...) {
        return false;
    }
}

void DFlashServer::handle_models(const httplib::Request& req, httplib::Response& res) {
    json response = {
        {"object", "list"},
        {"data", {
            {
                {"id", "dflash"},
                {"object", "model"},
                {"owned_by", "luce"},
                {"permission", {}}
            }
        }}
    };
    res.set_content(response.dump(), "application/json");
}

void DFlashServer::handle_chat_completions(const httplib::Request& req, httplib::Response& res) {
    try {
        ChatRequest chat_req = parse_chat_request(req);

        // Tokenize prompt
        std::vector<int32_t> prompt_tokens;
        // TODO: Apply chat template and tokenize

        // Check cache
        auto cached_slot = prefix_cache_->find_slot(prompt_tokens);

        std::vector<int32_t> tokens;

        if (cached_slot) {
            // Restore from cache
            // TODO: Implement cache restore
        } else {
            // Generate
            tokens = generate_internal(prompt_tokens, chat_req.max_tokens);

            // Cache result
            int slot = prefix_cache_->allocate_slot();
            if (slot >= 0) {
                // TODO: Snapshot cache
            }
        }

        // Build response
        ChatResponse chat_resp = build_chat_response(tokens, chat_req);

        // Convert to JSON
        json response = {
            {"id", chat_resp.id},
            {"object", chat_resp.object},
            {"created", chat_resp.created},
            {"model", chat_resp.model},
            {"choices", {
                {
                    {"index", 0},
                    {"message", {
                        {"role", "assistant"},
                        {"content", tokenizer_->decode(tokens)}
                    }},
                    {"finish_reason", "stop"}
                }
            }},
            {"usage", {
                {"prompt_tokens", static_cast<int>(prompt_tokens.size())},
                {"completion_tokens", static_cast<int>(tokens.size())},
                {"total_tokens", static_cast<int>(prompt_tokens.size() + tokens.size())}
            }}
        };

        res.set_content(response.dump(), "application/json");

    } catch (const std::exception& e) {
        json error = {{"error", {{"message", e.what()}}}};
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void DFlashServer::handle_chat_completions_stream(const httplib::Request& req, httplib::Response& res) {
    // For now, use non-streaming as placeholder
    // TODO: Implement proper SSE streaming with correct cpp-httplib API
    handle_chat_completions(req, res);
}

std::vector<int32_t> DFlashServer::generate_internal(
    const std::vector<int32_t>& prompt_tokens,
    int n_gen,
    std::function<void(int32_t)> on_token
) {
    std::lock_guard<std::mutex> lock(generation_mutex_);

    std::vector<int32_t> result;

    // TODO: Implement actual DFlash generation
    // This should call the existing generation logic from test_dflash.cpp
    //
    // Pseudocode:
    // 1. Prefill prompt tokens
    // 2. Decode loop:
    //    a. Draft forward (16 tokens)
    //    b. Target verify with DDTree
    //    c. Accept tokens
    //    d. Call on_token for each accepted token
    //    e. Replay accepted tokens

    return result;
}

ChatRequest DFlashServer::parse_chat_request(const httplib::Request& req) {
    json body = json::parse(req.body);

    ChatRequest req_data;
    req_data.model = body.value("model", "dflash");
    req_data.stream = body.value("stream", false);
    req_data.max_tokens = body.value("max_tokens", 50000);

    if (body.contains("messages")) {
        for (const auto& msg : body["messages"]) {
            ChatMessage m;
            m.role = msg["role"];
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    m.content = msg["content"];
                }
            }
            req_data.messages.push_back(m);
        }
    }

    // Parse other fields...
    return req_data;
}

ChatResponse DFlashServer::build_chat_response(
    const std::vector<int32_t>& tokens,
    const ChatRequest& req,
    const std::string& reasoning
) {
    ChatResponse resp;
    resp.id = "chatcmpl-" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    resp.created = std::time(nullptr);
    resp.model = req.model;

    ChatResponse::Choice choice;
    choice.message.role = "assistant";
    choice.message.content = tokenizer_->decode(tokens);
    choice.finish_reason = "stop";
    resp.choices.push_back(choice);

    resp.usage.prompt_tokens = 0;  // TODO
    resp.usage.completion_tokens = tokens.size();
    resp.usage.total_tokens = 0;  // TODO

    return resp;
}

// ─── Reasoning Extraction (using parsers) ────────────────────────────────

std::pair<std::string, std::string> DFlashServer::extract_reasoning(
    const std::string& text
) {
    // Use the parse_reasoning function from parsers.cpp
    ReasoningParseResult result = parse_reasoning(text, /*thinking_enabled=*/true);
    return {result.content, result.reasoning};
}

// ─── Anthropic Messages API Implementation ──────────────────────────────

AnthropicMessagesRequest DFlashServer::parse_anthropic_messages_request(
    const httplib::Request& req
) {
    json body = json::parse(req.body);

    AnthropicMessagesRequest areq;

    // Parse model
    areq.model = body.value("model", "dflash");

    // Parse stream flag
    areq.stream = body.value("stream", false);

    // Parse max_tokens (required by Anthropic)
    if (body.contains("max_tokens")) {
        areq.max_tokens = body["max_tokens"];
    }

    // Parse optional parameters
    if (body.contains("temperature")) {
        areq.temperature = body["temperature"];
    }
    if (body.contains("top_p")) {
        areq.top_p = body["top_p"];
    }
    if (body.contains("top_k")) {
        areq.top_k = body["top_k"];
    }

    // Parse system prompt
    if (body.contains("system")) {
        if (body["system"].is_string()) {
            areq.system = body["system"];
        }
        // TODO: Handle system as array of content blocks
    }

    // Parse stop_sequences
    if (body.contains("stop_sequences")) {
        std::vector<std::string> seqs;
        for (const auto& s : body["stop_sequences"]) {
            seqs.push_back(s);
        }
        areq.stop_sequences = std::move(seqs);
    }

    // Parse messages
    if (body.contains("messages")) {
        for (const auto& msg : body["messages"]) {
            AnthropicMessage am;
            am.role = msg["role"];

            // Parse content (can be string or array)
            if (msg.contains("content")) {
                if (msg["content"].is_string()) {
                    // Simple text content
                    AnthropicContentBlock block;
                    block.type = "text";
                    block.text = msg["content"];
                    am.content.push_back(block);
                } else if (msg["content"].is_array()) {
                    // Array of content blocks
                    for (const auto& cb : msg["content"]) {
                        AnthropicContentBlock block;
                        block.type = cb.value("type", "text");

                        if (block.type == "text") {
                            block.text = cb.value("text", "");
                        } else if (block.type == "image") {
                            // TODO: Handle image content
                            block.source_type = cb["source"].value("type", "");
                            block.media_type = cb["source"].value("media_type", "");
                            block.data = cb["source"].value("data", "");
                        } else if (block.type == "tool_use") {
                            block.id = cb.value("id", "");
                            block.input_json = cb.value("input", json::object()).dump();
                        } else if (block.type == "tool_result") {
                            block.id = cb.value("tool_use_id", "");
                            if (cb["content"].is_string()) {
                                block.text = cb["content"];
                            }
                        }

                        am.content.push_back(block);
                    }
                }
            }

            // Parse optional name
            if (msg.contains("name")) {
                am.name = msg["name"];
            }

            areq.messages.push_back(am);
        }
    }

    // Parse tools
    if (body.contains("tools")) {
        areq.tools = std::vector<AnthropicTool>{};
        for (const auto& tool : body["tools"]) {
            AnthropicTool atool;
            atool.name = tool["name"];
            atool.description = tool.value("description", "");
            atool.input_schema = tool["input_schema"];
            areq.tools->push_back(atool);
        }
    }

    // Parse tool_choice
    if (body.contains("tool_choice")) {
        auto& tc = body["tool_choice"];
        if (tc.is_string()) {
            std::string tc_str = tc;
            if (tc_str == "auto") {
                areq.tool_choice = AnthropicToolChoice{AnthropicToolChoiceType::auto_mode, ""};
            } else if (tc_str == "any") {
                areq.tool_choice = AnthropicToolChoice{AnthropicToolChoiceType::any, ""};
            }
        } else if (tc.is_object()) {
            std::string type = tc.value("type", "auto");
            if (type == "tool") {
                areq.tool_choice = AnthropicToolChoice{
                    AnthropicToolChoiceType::tool,
                    tc.value("name", "")
                };
            }
        }
    }

    return areq;
}

ChatRequest DFlashServer::convert_anthropic_to_chat(
    const AnthropicMessagesRequest& areq
) {
    ChatRequest creq;

    // Set basic fields
    creq.model = areq.model;
    creq.stream = areq.stream;
    creq.max_tokens = areq.max_tokens;
    creq.temperature = areq.temperature;
    creq.top_p = areq.top_p;

    // Convert messages
    for (const auto& am : areq.messages) {
        ChatMessage cm;
        cm.role = am.role;

        // Join all text content blocks
        std::string text_content;
        for (const auto& cb : am.content) {
            if (cb.type == "text") {
                text_content += cb.text;
            }
            // TODO: Handle tool_use and tool_result blocks
        }

        cm.content = text_content;

        if (am.name) {
            cm.name = *am.name;
        }

        creq.messages.push_back(cm);
    }

    // Handle system prompt (prepend as user message for now)
    // TODO: Properly integrate system prompt with chat template
    if (!areq.system.empty()) {
        ChatMessage sys_msg;
        sys_msg.role = "system";
        sys_msg.content = areq.system;
        creq.messages.insert(creq.messages.begin(), sys_msg);
    }

    // Convert stop_sequences to stop
    if (areq.stop_sequences) {
        creq.stop = *areq.stop_sequences;
    }

    // Convert tools (from Anthropic to OpenAI format)
    if (areq.tools) {
        creq.tools = std::vector<ToolDef>{};
        for (const auto& atool : *areq.tools) {
            ToolDef tdef;
            tdef.type = "function";
            tdef.function = {
                {"name", atool.name},
                {"description", atool.description},
                {"parameters", atool.input_schema}
            };
            creq.tools->push_back(tdef);
        }
    }

    return creq;
}

AnthropicMessagesResponse DFlashServer::build_anthropic_response(
    const std::vector<int32_t>& tokens,
    const AnthropicMessagesRequest& areq,
    const std::string& reasoning
) {
    AnthropicMessagesResponse resp;

    // Generate message ID
    auto now = std::chrono::system_clock::now();
    auto timestamp = now.time_since_epoch().count();
    resp.id = "msg_" + std::to_string(timestamp);

    resp.model = areq.model;
    resp.type = "message";
    resp.role = "assistant";

    // Decode tokens to text
    std::string text = tokenizer_->decode(tokens);

    // Parse reasoning if present
    std::string content_text = text;
    if (!reasoning.empty()) {
        // If reasoning was provided separately, use it
        // TODO: Handle reasoning content in Anthropic format
    } else {
        // Check if text contains reasoning tags
        size_t think_pos = text.find("\xe5\x8f\x8d\xe5\xa4\x8d\xe6\x8e\xa8\xe7\x90\x86");  // 反复推理
        if (think_pos != std::string::npos) {
            // Has reasoning - extract it
            auto [content, reasoning_content] = extract_reasoning(text);
            content_text = content;
            // TODO: Include reasoning in response
        }
    }

    // Create text content block
    AnthropicContentBlock block;
    block.type = "text";
    block.text = content_text;
    resp.content.push_back(block);

    // Set stop_reason
    resp.stop_reason = "end_turn";  // Default
    // TODO: Determine actual stop_reason from generation

    // Set usage
    resp.usage.input_tokens = 0;  // TODO: Get from prompt
    resp.usage.output_tokens = tokens.size();

    return resp;
}

json DFlashServer::anthropic_response_to_json(
    const AnthropicMessagesResponse& resp
) {
    json content_blocks = json::array();
    for (const auto& cb : resp.content) {
        json block;
        block["type"] = cb.type;

        if (cb.type == "text") {
            block["text"] = cb.text;
        } else if (cb.type == "tool_use") {
            block["id"] = cb.id;
            block["name"] = cb.text;  // Reusing text field for tool name
            block["input"] = json::parse(cb.input_json);
        } else if (cb.type == "tool_result") {
            block["tool_use_id"] = cb.id;
            block["content"] = cb.text;
        } else if (cb.type == "image") {
            block["source"] = {
                {"type", cb.source_type},
                {"media_type", cb.media_type},
                {"data", cb.data}
            };
        }

        content_blocks.push_back(block);
    }

    json response = {
        {"id", resp.id},
        {"type", resp.type},
        {"role", resp.role},
        {"content", content_blocks},
        {"model", resp.model},
        {"stop_reason", resp.stop_reason},
        {"usage", {
            {"input_tokens", resp.usage.input_tokens},
            {"output_tokens", resp.usage.output_tokens}
        }}
    };

    return response;
}

std::string DFlashServer::anthropic_event_to_sse(
    const std::string& event_name,
    const json& event_data
) {
    std::ostringstream oss;
    oss << "event: " << event_name << "\n";
    oss << "data: " << event_data.dump() << "\n\n";
    return oss.str();
}

void DFlashServer::handle_anthropic_messages(
    const httplib::Request& req,
    httplib::Response& res
) {
    try {
        AnthropicMessagesRequest areq = parse_anthropic_messages_request(req);

        if (areq.stream) {
            handle_anthropic_messages_stream_impl(areq, res);
        } else {
            handle_anthropic_messages_impl(areq, res);
        }

    } catch (const std::exception& e) {
        // Return Anthropic-style error
        json error = {
            {"type", "error"},
            {"error", {
                {"type", "internal_error"},
                {"message", std::string("Internal error: ") + e.what()}
            }}
        };
        res.status = 500;
        res.set_content(error.dump(), "application/json");
    }
}

void DFlashServer::handle_anthropic_messages_impl(
    const AnthropicMessagesRequest& areq,
    httplib::Response& res
) {
    // Convert to ChatRequest and use existing generation logic
    ChatRequest creq = convert_anthropic_to_chat(areq);

    // Tokenize prompt
    std::vector<int32_t> prompt_tokens;
    // TODO: Apply chat template and tokenize

    // Check cache
    auto cached_slot = prefix_cache_->find_slot(prompt_tokens);

    std::vector<int32_t> tokens;

    if (cached_slot) {
        // Restore from cache
        // TODO: Implement cache restore
    } else {
        // Generate
        tokens = generate_internal(prompt_tokens, creq.max_tokens);

        // Cache result
        int slot = prefix_cache_->allocate_slot();
        if (slot >= 0) {
            // TODO: Snapshot cache
        }
    }

    // Build Anthropic response
    AnthropicMessagesResponse aresp = build_anthropic_response(tokens, areq);

    // Convert to JSON
    json response_json = anthropic_response_to_json(aresp);

    res.set_content(response_json.dump(), "application/json");
}

void DFlashServer::handle_anthropic_messages_stream_impl(
    const AnthropicMessagesRequest& areq,
    httplib::Response& res
) {
    // Set SSE headers
    res.set_header("Content-Type", "text/event-stream");
    res.set_header("Cache-Control", "no-cache");
    res.set_header("Connection", "keep-alive");

    // For now, use non-streaming implementation as placeholder
    // TODO: Implement proper SSE streaming with Anthropic event format
    handle_anthropic_messages_impl(areq, res);
}

void DFlashServer::run() {
    running_ = true;
    std::cout << "DFlash server starting on " << config_.host << ":" << config_.port << std::endl;
    http_server_->listen(config_.host, config_.port);
}

void DFlashServer::stop() {
    running_ = false;
    if (http_server_) {
        http_server_->stop();
    }
}

// ─── Helper Functions ───────────────────────────────────────────────────

std::string tokens_to_text(const std::vector<int32_t>& tokens, Tokenizer* tokenizer) {
    if (!tokenizer || tokens.empty()) return "";
    return tokenizer->decode(tokens);
}

// ─── Main Entry Point ───────────────────────────────────────────────────

} // namespace dflash27b

// ─── Standalone Server Main Function ─────────────────────────────────────

int main(int argc, char** argv) {
    using namespace dflash27b;

    // Parse command line arguments
    ServerConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--target" && i + 1 < argc) {
            config.target_path = argv[++i];
        } else if (arg == "--draft" && i + 1 < argc) {
            config.draft_path = argv[++i];
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::atoi(argv[++i]);
        } else if (arg == "--host" && i + 1 < argc) {
            config.host = argv[++i];
        } else if (arg == "--max-ctx" && i + 1 < argc) {
            config.max_ctx = std::atoi(argv[++i]);
        } else if (arg == "--prefill-mode" && i + 1 < argc) {
            config.prefill_mode = argv[++i];
        } else if (arg == "--prefill-always") {
            config.prefill_mode = "always";
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " --target <gguf> --draft <safetensors> [options]\n"
                      << "\nOptions:\n"
                      << "  --target <path>    Target model GGUF file\n"
                      << "  --draft <path>     Draft model safetensors file\n"
                      << "  --host <addr>      Server host (default: 0.0.0.0)\n"
                      << "  --port <port>      Server port (default: 8000)\n"
                      << "  --max-ctx <n>      Max context length (default: 131072)\n"
                      << "  --prefill-mode     off/auto/always (default: auto)\n"
                      << "  --prefill-always    Alias for --prefill-mode always\n";
            return 0;
        }
    }

    // Validate required arguments
    if (config.target_path.empty()) {
        std::cerr << "Error: --target is required" << std::endl;
        return 1;
    }
    if (config.draft_path.empty()) {
        std::cerr << "Error: --draft is required" << std::endl;
        return 1;
    }

    // Create and initialize server
    DFlashServer server;
    if (!server.initialize(config)) {
        std::cerr << "Failed to initialize server" << std::endl;
        return 1;
    }

    // Run server
    server.run();

    return 0;
}
