// dflash_server.cpp - C++ HTTP Server Implementation
//
// This file implements a standalone HTTP server for DFlash that replaces
// the Python-based server_tools.py.

#include "dflash_server.h"
#include "internal.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include "cpp-httplib/httplib.h"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace dflash27b {

// ─── Tokenizer Wrapper (using llama.cpp) ───────────────────────────────

class Tokenizer {
public:
    bool load_from_gguf(const std::string& gguf_path) {
        // TODO: Use llama.cpp's tokenizer
        // For now, placeholder
        return true;
    }

    std::vector<int32_t> encode(const std::string& text) {
        // TODO: Implement using llama.cpp
        return {};
    }

    std::string decode(const std::vector<int32_t>& tokens) {
        // TODO: Implement using llama.cpp
        return "";
    }
};

// ─── Chat Template Engine ───────────────────────────────────────────────

class ChatTemplate {
public:
    bool load_from_gguf(const std::string& gguf_path) {
        // TODO: Use llama.cpp's Jinja2 runtime
        return true;
    }

    std::string apply(const std::vector<ChatMessage>& messages) {
        // TODO: Implement using llama.cpp's jinja engine
        return "";
    }
};

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
        std::cerr << "Failed to load tokenizer" << std::endl;
        return false;
    }

    // Initialize chat template
    chat_template_ = std::make_unique<ChatTemplate>();
    if (!chat_template_->load_from_gguf(config.target_path)) {
        std::cerr << "Failed to load chat template" << std::endl;
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

    // TODO: Load target and draft models
    // TODO: Initialize KV cache

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
