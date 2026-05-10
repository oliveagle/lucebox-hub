#include "tokenizer_parser.h"
#include "tokenizer.h"

#include <sstream>

namespace dflash27b {

// ─── Tokenization Helper Utilities ───────────────────────────────────────

std::vector<int32_t> encode_role_prefix(Tokenizer* tokenizer, const std::string& role) {
    // ChatML format: <|im_start|>{role}\n
    std::string prefix = "<|im_start|>" + role + "\n";
    return tokenizer->encode(prefix);
}

std::vector<int32_t> encode_eom(Tokenizer* tokenizer) {
    // ChatML format: <|im_end|>
    return tokenizer->encode("<|im_end|>");
}

std::vector<int32_t> encode_chatml_message(
    Tokenizer* tokenizer,
    const std::string& role,
    const std::string& content
) {
    // ChatML format: <|im_start|>{role}\n{content}<|im_end|>
    std::string message = "<|im_start|>" + role + "\n" + content + "<|im_end|>";
    return tokenizer->encode(message);
}

std::string tokens_debug_repr(const std::vector<int32_t>& tokens) {
    if (tokens.empty()) return "[]";

    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << tokens[i];
    }
    oss << "]";
    return oss.str();
}

int find_token_sequence(
    const std::vector<int32_t>& haystack,
    const std::vector<int32_t>& needle,
    int start_index
) {
    if (needle.empty()) return start_index;
    if (needle.size() > haystack.size()) return -1;

    for (size_t i = static_cast<size_t>(start_index); i <= haystack.size() - needle.size(); ++i) {
        bool match = true;
        for (size_t j = 0; j < needle.size(); ++j) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return static_cast<int>(i);
    }

    return -1;
}

} // namespace dflash27b
