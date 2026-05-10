// tokenizer_parser.h - Parser for dflash server token output
//
// Parses lines like:
//   151644: </s>
//   8948: user
//   198: \n
//
// Also provides helper utilities for tokenization patterns.

#pragma once

#include <cstdint>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace dflash27b {

// Forward declaration
class Tokenizer;

struct TokenInfo {
    int32_t id;
    std::string text;
};

class TokenizerParser {
public:
    TokenizerParser() {
        // Regex: optional [DEBUG] prefix, token_id, colon, text
        pattern_ = R"((?:\[DEBUG\])?\s*(\d+):\s*(.*))";
    }

    std::optional<TokenInfo> parse_line(const std::string& line) {
        std::smatch match;
        if (std::regex_search(line, match, pattern_)) {
            TokenInfo info;
            info.id = std::stoi(match[1].str());
            info.text = match[2].str();
            return info;
        }
        return std::nullopt;
    }

    std::vector<TokenInfo> parse_output(const std::string& output) {
        std::vector<TokenInfo> tokens;
        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) {
            auto token = parse_line(line);
            if (token) {
                tokens.push_back(*token);
            }
        }
        return tokens;
    }

private:
    std::regex pattern_;
};

// ─── Tokenization Helper Utilities ───────────────────────────────────────

// Encode a message role prefix (e.g., "<|im_start|>user\n").
std::vector<int32_t> encode_role_prefix(Tokenizer* tokenizer, const std::string& role);

// Encode the end-of-message marker (e.g., "<|im_end|>").
std::vector<int32_t> encode_eom(Tokenizer* tokenizer);

// Encode a message with the ChatML format:
//   <|im_start|>{role}\n{content}<|im_end|>
std::vector<int32_t> encode_chatml_message(
    Tokenizer* tokenizer,
    const std::string& role,
    const std::string& content
);

// Tokenize text into a human-readable form for debugging.
// Returns a string like "[151644, 151646, 2718, ...]".
std::string tokens_debug_repr(const std::vector<int32_t>& tokens);

// Find the first occurrence of a token sequence in a token stream.
// Returns the index of the first match, or -1 if not found.
int find_token_sequence(
    const std::vector<int32_t>& haystack,
    const std::vector<int32_t>& needle,
    int start_index = 0
);

} // namespace dflash27b
