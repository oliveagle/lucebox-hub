#include "tokenizer.h"

#include "llama.h"

#include <cstring>
#include <vector>

namespace dflash27b {

Tokenizer::~Tokenizer() {
    if (ctx_) {
        llama_free(ctx_);
    }
    if (model_) {
        llama_model_free(const_cast<llama_model*>(model_));
    }
}

bool Tokenizer::load_from_gguf(const std::string& gguf_path) {
    // Clean up any previous state.
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(const_cast<llama_model*>(model_));
        model_ = nullptr;
    }

    // Initialize llama.cpp (no-op if already initialized).
    llama_backend_init();

    // Load model parameters.
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;  // Tokenizer runs entirely on CPU.

    model_ = llama_model_load_from_file(gguf_path.c_str(), mparams);
    if (!model_) {
        last_error_ = "llama_model_load_from_file failed: " + gguf_path;
        return false;
    }

    // Create context (minimal, we only need tokenizer functionality).
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 8192;
    cparams.offload_kqv = false;
    cparams.embeddings  = false;

    ctx_ = llama_init_from_model(const_cast<llama_model*>(model_), cparams);
    if (!ctx_) {
        llama_model_free(const_cast<llama_model*>(model_));
        model_ = nullptr;
        last_error_ = "llama_init_from_model failed";
        return false;
    }

    return true;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    if (!ctx_) return {};

    // Allocate a buffer large enough (4 chars per token is a safe upper bound).
    int32_t n = static_cast<int32_t>(text.size() + 2);
    std::vector<int32_t> tokens(n);

    // Use llama_tokenize with special tokens as strings (not converted to tokens).
    int result = llama_tokenize(
        llama_model_get_vocab(llama_get_model(const_cast<llama_context*>(ctx_))),
        text.c_str(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        /* add_special */ false,
        /* parse_special */ false
    );

    if (result < 0) {
        // Realloc with larger buffer and retry.
        tokens.resize(-result);
        result = llama_tokenize(
            llama_model_get_vocab(llama_get_model(const_cast<llama_context*>(ctx_))),
            text.c_str(),
            static_cast<int32_t>(text.size()),
            tokens.data(),
            static_cast<int32_t>(tokens.size()),
            /* add_special */ false,
            /* parse_special */ false
        );
        if (result < 0) return {};
    }

    tokens.resize(static_cast<size_t>(result));
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    if (!ctx_ || tokens.empty()) return "";

    std::string result;
    result.reserve(tokens.size() * 4);  // Heuristic reservation.

    for (int32_t token_id : tokens) {
        char buf[256];
        int n = decode_token_to_buffer(token_id, buf, static_cast<int>(sizeof(buf)));
        if (n > 0) {
            result.append(buf, static_cast<size_t>(n));
        }
    }

    return result;
}

std::string Tokenizer::decode_token(int32_t token_id) const {
    char buf[256];
    int n = decode_token_to_buffer(token_id, buf, static_cast<int>(sizeof(buf)));
    if (n <= 0) return "";
    return std::string(buf, static_cast<size_t>(n));
}

int Tokenizer::decode_token_to_buffer(int32_t token_id, char* buffer, int max_buffer_len) const {
    if (!ctx_ || max_buffer_len <= 0) return 0;

    int n = llama_token_to_piece(
        llama_model_get_vocab(llama_get_model(const_cast<llama_context*>(ctx_))),
        token_id,
        buffer,
        max_buffer_len,
        0,  // lstrip
        false  // special
    );

    if (n < 0) {
        // Buffer too small, return 0.
        return 0;
    }
    return n;
}

int32_t Tokenizer::vocab_size() const {
    if (!model_) return 0;
    return llama_vocab_n_tokens(llama_model_get_vocab(model_));
}

int32_t Tokenizer::bos_token() const {
    if (!model_) return -1;
    return llama_vocab_bos(llama_model_get_vocab(model_));
}

int32_t Tokenizer::eos_token() const {
    if (!model_) return -1;
    return llama_vocab_eos(llama_model_get_vocab(model_));
}

int32_t Tokenizer::eot_token() const {
    if (!model_) return -1;
    return llama_vocab_eot(llama_model_get_vocab(model_));
}

} // namespace dflash27b
