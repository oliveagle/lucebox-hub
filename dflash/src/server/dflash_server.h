// dflash_server.h - C++ HTTP Server for DFlash
//
// This file defines a standalone HTTP server that implements all the
// functionality previously in server_tools.py, using only C++.

#pragma once

#include <cstdint>
#include <optional>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// cpp-httplib is in llama.cpp/vendor
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

// ggml types (forward declarations)
struct ggml_context;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_gallocr;

namespace dflash27b {

// Forward declarations
struct TargetWeights;
struct DraftWeights;
struct TargetCache;
struct DeltaNetCapture;
class Tokenizer;
class PrefixCache;
class PrefillCoordinator;
struct DrafterContext;

// ─── Configuration ─────────────────────────────────────────────────────

struct ServerConfig {
    std::string target_path;
    std::string draft_path;
    std::string host = "0.0.0.0";
    int port = 8000;
    int max_ctx = 131072;
    int ddtree_budget = 22;
    bool fast_rollback = true;

    // PFlash compression
    bool prefill_enabled = true;
    std::string prefill_mode = "always";  // "off", "auto", "always"
    int prefill_threshold = 8192;
    float prefill_keep_ratio = 0.1f;
    std::string prefill_drafter;

    // KV cache
    std::string kv_k_type = "q8_0";
    std::string kv_v_type = "q8_0";
    int fa_window = 2048;
};

// ─── Request/Response Types ───────────────────────────────────────────

struct ToolCallFunction {
    std::string name;
    std::string arguments;  // JSON string
};

struct ToolCall {
    std::string id;
    std::string type = "function";
    ToolCallFunction function;
};

struct ChatMessage {
    std::string role;
    std::string content;  // Simplified (could be object/array)
    std::optional<std::string> name;
    std::optional<std::string> tool_call_id;
    std::optional<std::vector<ToolCall>> tool_calls;
};

struct ToolDef {
    std::string type = "function";
    nlohmann::json function;  // {name, description, parameters}
};

struct ChatRequest {
    std::string model = "dflash";
    std::vector<ChatMessage> messages;
    bool stream = false;
    int max_tokens = 50000;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::vector<ToolDef>> tools;
    std::optional<nlohmann::json> tool_choice;
    std::optional<nlohmann::json> chat_template_kwargs;
    std::optional<nlohmann::json> stop;
    std::optional<nlohmann::json> stream_options;
};

struct ChatResponse {
    std::string id;
    std::string object = "chat.completion";
    int created;
    std::string model;
    struct Choice {
        int index = 0;
        ChatMessage message;
        std::string finish_reason;
    };
    std::vector<Choice> choices;
    struct Usage {
        int prompt_tokens;
        int completion_tokens;
        int total_tokens;
    } usage;
};

// ─── Anthropic Messages API Types ─────────────────────────────────────

// Anthropic content block (text, image, tool_use, tool_result)
struct AnthropicContentBlock {
    std::string type;  // "text", "image", "tool_use", "tool_result"
    std::string text;
    // For tool_use
    std::string id;
    std::string input_json;
    // For image (not implemented yet)
    std::string source_type;
    std::string media_type;
    std::string data;
};

// Anthropic message (role + content blocks)
struct AnthropicMessage {
    std::string role;  // "user" or "assistant"
    std::vector<AnthropicContentBlock> content;  // Can be text, images, etc.
    std::optional<std::string> name;  // Optional name for the message
};

// Anthropic tool definition
struct AnthropicTool {
    std::string name;
    std::string description;
    nlohmann::json input_schema;  // JSON Schema
};

// Anthropic tool_choice
enum class AnthropicToolChoiceType {
    auto_mode,   // default
    any,
    tool         // specific tool
};

struct AnthropicToolChoice {
    AnthropicToolChoiceType type = AnthropicToolChoiceType::auto_mode;
    std::string tool_name;  // only for tool type
};

// Anthropic Messages API request
struct AnthropicMessagesRequest {
    std::string model = "dflash";
    std::vector<AnthropicMessage> messages;
    // System prompt can be string or array of content blocks
    std::string system;
    bool stream = false;
    int max_tokens = 4096;
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<float> top_k;
    std::optional<std::vector<std::string>> stop_sequences;
    std::optional<std::vector<AnthropicTool>> tools;
    std::optional<AnthropicToolChoice> tool_choice;
};

// Anthropic Messages API response (non-streaming)
struct AnthropicMessagesResponse {
    std::string id;
    std::string model;
    std::string type = "message";
    std::string role = "assistant";
    std::vector<AnthropicContentBlock> content;
    std::string stop_reason;  // "end_turn", "max_tokens", "stop_sequence", "tool_use"
    struct Usage {
        int input_tokens;
        int output_tokens;
    } usage;
};

// ─── Graph State ──────────────────────────────────────────────────────────

// StepGraph: persistent graph state reused across forward calls
struct StepGraph {
    ggml_context * ctx = nullptr;
    ggml_cgraph * gf = nullptr;
    ggml_gallocr * alloc = nullptr;
    ggml_tensor * inp_embed = nullptr;    // [hidden, n_tokens, 1] f32
    ggml_tensor * positions = nullptr;    // [4*n_tokens] i32
    ggml_tensor * attn_mask = nullptr;    // [kv_pad, q_pad] f16
    ggml_tensor * parent_ids = nullptr;   // [n_tokens] i32 (DDTree only)
    ggml_tensor * logits = nullptr;       // [vocab, n_tokens] f32
    ggml_tensor * argmax_tokens = nullptr; // [n_tokens] i32
    ggml_tensor * hidden_states = nullptr; // [hidden, n_tokens] f32 (draft)
    ggml_tensor * hidden_input = nullptr;   // [hidden, n_tokens] f32 (lm_head proj)
    std::vector<dflash27b::DeltaNetCapture> delta_captures;
};

// DraftFeatureMirror: rolling buffer of captured target features
struct DraftFeatureMirror {
    ggml_tensor * target_feat = nullptr;  // [N*hidden, cap] bf16
    int cap = 0;
    int offset = 0;  // target_cache.target_feat offset
};

// ─── Main Server Class ─────────────────────────────────────────────────

class DFlashServer {
public:
    DFlashServer();
    ~DFlashServer();

    // Initialize the server with configuration
    bool initialize(const ServerConfig& config);

    // Run the HTTP server (blocking call)
    void run();

    // Stop the server
    void stop();

    // Route handlers
    void setup_routes();

    void handle_models(const httplib::Request& req, httplib::Response& res);
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_chat_completions_stream(const httplib::Request& req, httplib::Response& res);
    void handle_anthropic_messages(const httplib::Request& req, httplib::Response& res);

    // Helper to check if request is streaming
    static bool is_streaming_request(const httplib::Request& req);

    // Core generation logic
    std::vector<int32_t> generate_internal(
        const std::vector<int32_t>& prompt_tokens,
        int n_gen,
        std::function<void(int32_t)> on_token = nullptr
    );

    // Request parsing
    ChatRequest parse_chat_request(const httplib::Request& req);
    ChatResponse build_chat_response(
        const std::vector<int32_t>& tokens,
        const ChatRequest& req,
        const std::string& reasoning = ""
    );

    // Tool call parsing
    std::string parse_tool_calls(
        const std::string& text,
        const std::vector<ToolDef>* tools
    );

    // Reasoning extraction
    std::pair<std::string, std::string> extract_reasoning(const std::string& text);

    // ─── Anthropic Messages API Methods ─────────────────────────────

    // Parse Anthropic Messages API request from JSON body
    AnthropicMessagesRequest parse_anthropic_messages_request(const httplib::Request& req);

    // Convert Anthropic request to internal ChatRequest format
    ChatRequest convert_anthropic_to_chat(const AnthropicMessagesRequest& areq);

    // Build Anthropic response from generated tokens
    AnthropicMessagesResponse build_anthropic_response(
        const std::vector<int32_t>& tokens,
        const AnthropicMessagesRequest& areq,
        const std::string& reasoning = ""
    );

    // Serialize Anthropic response to JSON
    nlohmann::json anthropic_response_to_json(const AnthropicMessagesResponse& resp);

    // Serialize Anthropic streaming event to SSE format
    std::string anthropic_event_to_sse(
        const std::string& event_name,
        const nlohmann::json& event_data
    );

    // Handle Anthropic Messages API (non-streaming)
    void handle_anthropic_messages_impl(
        const AnthropicMessagesRequest& areq,
        httplib::Response& res
    );

    // Handle Anthropic Messages API (streaming)
    void handle_anthropic_messages_stream_impl(
        const AnthropicMessagesRequest& areq,
        httplib::Response& res
    );

    // ─── PFlash Compression ────────────────────────────────────────────────

    // Load the PFlash drafter model (lazy loading, called on first compress)
    bool load_drafter();

    // Free the PFlash drafter to free VRAM
    void free_drafter();

    // Compress prompt tokens using PFlash drafter
    // Returns compressed token IDs (drafter vocab)
    std::vector<int32_t> compress_prefill(const std::vector<int32_t>& prompt_tokens);

    // ─── DFlash Generation ─────────────────────────────────────────────

    // Graph step builders
    bool build_target_step(StepGraph& sg, int kv_start, int n_tokens,
                           bool with_mask, bool capture,
                           bool capture_delta_intermediate = false,
                           int fa_window = 0);
    bool build_target_step_tree(StepGraph& sg, int kv_start, int n_tokens, int fa_window = 0);
    bool build_draft_step(StepGraph& sg, int draft_ctx,
                          DraftFeatureMirror* mirror, int committed);
    bool build_lm_head_projection_step(StepGraph& sg, int q_len);
    void step_graph_destroy(StepGraph& sg);

    // DFlash decode loop
    bool run_dflash_decode(
        const std::vector<int32_t>& prompt,
        int n_gen,
        int last_tok,
        std::function<void(int32_t)> on_token,
        int& n_generated);

    // Configuration
    ServerConfig config_;

    // HTTP server
    std::unique_ptr<httplib::Server> http_server_;

    // Core components
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    std::unique_ptr<PrefillCoordinator> prefill_;

    // DFlash model state
    std::unique_ptr<TargetWeights> target_weights_;
    std::unique_ptr<DraftWeights> draft_weights_;
    std::unique_ptr<TargetCache> cache_;

    // PFlash drafter (for prefill compression)
    std::unique_ptr<DrafterContext> drafter_ctx_;
    bool drafter_loaded_ = false;

    // DFlash graph steps (reused across calls)
    StepGraph target_sg_;
    StepGraph draft_sg_;
    StepGraph proj_sg_;
    DraftFeatureMirror feature_mirror_;

    // Thread safety
    std::mutex generation_mutex_;
    bool running_ = false;
};

// ─── Helper Functions ───────────────────────────────────────────────────

// Convert tokens to text using tokenizer
std::string tokens_to_text(const std::vector<int32_t>& tokens, Tokenizer* tokenizer);

} // namespace dflash27b
