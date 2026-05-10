#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <optional>

#include "nlohmann/json.hpp"

struct llama_model;
struct llama_context;
struct llama_vocab;

namespace dflash27b {

// Forward declarations (defined in dflash_server.h)
struct ChatMessage;
struct ToolCall;
struct ToolDef;

// Tokenizer wrapper using llama.cpp's tokenizer infrastructure.
// Loads tokenizer metadata from a GGUF file and provides encode/decode.
class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();

    // Non-copyable
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    // Load tokenizer from a GGUF file.
    // Returns true on success, false on failure (error message in last_error_).
    bool load_from_gguf(const std::string& gguf_path);

    // Encode text to token IDs.
    std::vector<int32_t> encode(const std::string& text) const;

    // Decode token IDs to text.
    std::string decode(const std::vector<int32_t>& tokens) const;

    // Decode a single token to string (used by streaming).
    std::string decode_token(int32_t token_id) const;

    // Decode a single token to a buffer.
    // Returns the number of bytes written (may be 0 for special tokens).
    // The buffer should be at least `max_buffer_len` bytes.
    int decode_token_to_buffer(int32_t token_id, char* buffer, int max_buffer_len) const;

    // Get the vocabulary size.
    int32_t vocab_size() const;

    // Get special token IDs.
    int32_t bos_token() const;
    int32_t eos_token() const;
    int32_t eot_token() const;

    // Get the last error message.
    const std::string& last_error() const { return last_error_; }

    // Apply chat template to a list of messages.
    // Returns the formatted prompt string (not tokenized).
    // If chat_template is nullptr, uses the model's default template.
    std::string apply_chat_template(
        const std::vector<ChatMessage>& messages,
        const char* chat_template = nullptr,
        bool add_generation_prompt = true,
        const std::vector<ToolDef>* tools = nullptr,
        const nlohmann::json* chat_template_kwargs = nullptr
    ) const;

private:
    // llama.cpp handles
    const llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;

    // Error message for failed operations.
    mutable std::string last_error_;
};

} // namespace dflash27b
