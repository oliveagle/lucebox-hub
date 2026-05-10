// dflash_server.h - C++ HTTP Server for DFlash
//
// This file defines a standalone HTTP server that implements all the
// functionality previously in server_tools.py, using only C++.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// cpp-httplib is in llama.cpp/vendor
#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

namespace dflash27b {

// Forward declarations
struct TargetWeights;
struct DraftWeights;
struct TargetCache;
class Tokenizer;
class ChatTemplate;
class PrefixCache;
class PrefillCoordinator;

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

    // Configuration
    ServerConfig config_;

    // HTTP server
    std::unique_ptr<httplib::Server> http_server_;

    // Core components
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<ChatTemplate> chat_template_;
    std::unique_ptr<PrefixCache> prefix_cache_;
    std::unique_ptr<PrefillCoordinator> prefill_;

    // DFlash model state
    std::unique_ptr<TargetWeights> target_weights_;
    std::unique_ptr<DraftWeights> draft_weights_;
    std::unique_ptr<TargetCache> cache_;

    // Thread safety
    std::mutex generation_mutex_;
    bool running_ = false;
};

// ─── Helper Functions ───────────────────────────────────────────────────

// Convert tokens to text using tokenizer
std::string tokens_to_text(const std::vector<int32_t>& tokens, Tokenizer* tokenizer);

// Apply chat template to get prompt tokens
std::vector<int32_t> apply_chat_template(
    const std::vector<ChatMessage>& messages,
    ChatTemplate* chat_template,
    Tokenizer* tokenizer,
    bool add_generation_prompt = true
);

} // namespace dflash27b
