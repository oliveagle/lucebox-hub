// parsers.cpp - Tool call and reasoning parsers implementation
//
// Ported from vLLM's Qwen3CoderToolParser and Qwen3ReasoningParser
// (Apache-2.0 license)

#include "parsers.h"
#include "dflash_server.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <regex>
#include <sstream>
#include <random>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace dflash27b {

// ─── Constants ─────────────────────────────────────────────────────────────

// Qwen3 tool call XML tags
constexpr const char* TOOL_OPEN_TAG = "\xe8\xae\xbf\xe9\x97\xae\xe5\x87\xbd\xe6\x95\xb0";  // 访问函数
constexpr const char* THINK_OPEN_TAG = "\xe5\x8f\x8d\xe5\xa4\x8d\xe6\x8e\xa8\xe7\x90\x86";    // 反复推理
constexpr const char* THINK_CLOSE_TAG = "\xe7\xbb\x93\xe6\x9d\x9f\xe6\x8e\xa8\xe7\x90\x86";  // 结束推理

// Regex patterns (compiled on first use)
// Note: C++ std::regex doesn't support dotall mode, so we use [\s\S] instead of .
std::regex* get_tool_call_complete_re() {
    static std::regex re(R"(\xe8\xae\xbf\xe9\x97\xae\xe5\x87\xbd\xe6\x95\xb0([\s\S]*?)\xe7\xbb\x93\xe6\x9d\x9f\xe8\xae\xbf\xe9\x97\xae)");
    return &re;
}

std::regex* get_tool_call_function_re() {
    static std::regex re(R"(<function=(.*?)</function>|<function=(.*)$)");
    return &re;
}

std::regex* get_tool_call_parameter_re() {
    // vLLM's improved parameter regex: tolerates unclosed </parameter>
    // Note: Using [\s\S] instead of . for matching across newlines
    static std::regex re(R"(<parameter=([\s\S]*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$))");
    return &re;
}

// ─── Helper Functions ───────────────────────────────────────────────────────

std::string generate_call_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);

    std::string id = "call_";
    for (int i = 0; i < 24; ++i) {
        id += "0123456789abcdef"[dis(gen)];
    }
    return id;
}

nlohmann::json find_tool_properties(
    const std::vector<ToolDef>* tools,
    const std::string& function_name
) {
    if (!tools) return json::object();

    for (const auto& t : *tools) {
        const json& fn = t.function;
        if (fn.contains("name") && fn["name"] == function_name) {
            if (fn.contains("parameters") && fn["parameters"].is_object()) {
                return fn["parameters"].value("properties", json::object());
            }
        }
    }
    return json::object();
}

std::string convert_param_value(
    const std::string& param_value,
    const std::string& param_name,
    const nlohmann::json& param_config,
    const std::string& function_name
) {
    // Coerce stringified XML values to their JSON-schema type

    // Check for null
    std::string lower_value = param_value;
    for (auto& c : lower_value) c = std::tolower(c);
    if (lower_value == "null") {
        return "null";
    }

    // If no param config, return as string
    if (!param_config.contains(param_name)) {
        json j = param_value;
        return j.dump();
    }

    const auto& cfg = param_config[param_name];

    // Determine type
    std::string ptype;
    if (cfg.contains("type")) {
        ptype = cfg["type"].get<std::string>();
    } else if (cfg.contains("anyOf")) {
        ptype = "object";
    } else {
        ptype = "string";
    }

    // Convert based on type
    for (auto& c : ptype) c = std::tolower(c);

    if (ptype == "string" || ptype == "str" || ptype == "text" ||
        ptype == "varchar" || ptype == "char" || ptype == "enum") {
        json j = param_value;
        return j.dump();
    }

    if (ptype.find("int") == 0 || ptype.find("uint") == 0 ||
        ptype.find("long") == 0 || ptype.find("short") == 0 ||
        ptype.find("unsigned") == 0) {
        try {
            int val = std::stoi(param_value);
            return json(val).dump();
        } catch (...) {
            json j = param_value;
            return j.dump();
        }
    }

    if (ptype.find("num") == 0 || ptype.find("float") == 0) {
        try {
            double f = std::stod(param_value);
            // Return as int if it's a whole number
            if (f - std::floor(f) == 0.0) {
                return json(static_cast<int>(f)).dump();
            }
            return json(f).dump();
        } catch (...) {
            json j = param_value;
            return j.dump();
        }
    }

    if (ptype == "boolean" || ptype == "bool" || ptype == "binary") {
        bool val = (lower_value == "true");
        return json(val).dump();
    }

    // object / array / dict / list
    if (ptype == "object" || ptype == "array" || ptype == "arr" ||
        ptype.find("dict") == 0 || ptype.find("list") == 0) {
        try {
            json j = json::parse(param_value);
            return j.dump();
        } catch (...) {
            // Fall through to literal_eval attempt
        }
    }

    // Try literal_eval for Python-style literals
    // Simple cases: numbers, booleans, strings
    if (param_value == "true" || param_value == "True") {
        return "true";
    }
    if (param_value == "false" || param_value == "False") {
        return "false";
    }

    // Default: return as JSON string
    json j = param_value;
    return j.dump();
}

// ─── Tool Call Parsing ─────────────────────────────────────────────────────

ToolCallParseResult parse_tool_calls(
    const std::string& text,
    const std::vector<ToolDef>* tools
) {
    ToolCallParseResult result;
    std::vector<ParsedToolCall> tool_calls;
    std::vector<std::string> cleaned_parts;

    size_t cursor = 0;
    std::sregex_iterator iter(text.begin(), text.end(), *get_tool_call_complete_re());
    std::sregex_iterator iter_end;

    for (std::sregex_iterator it = iter; it != iter_end; ++it) {
        std::smatch match = *it;

        // Add content before this tool call
        cleaned_parts.push_back(text.substr(cursor, match.position() - cursor));
        cursor = match.position() + match.length();

        std::string body = match[1].str();

        // Extract function name
        std::smatch fn_match;
        std::string function_name;
        std::string params_region;

        if (std::regex_search(body, fn_match, *get_tool_call_function_re())) {
            std::string fn_text = fn_match[1].matched ? fn_match[1].str() : fn_match[2].str();

            size_t end_idx = fn_text.find('>');
            if (end_idx != std::string::npos) {
                function_name = fn_text.substr(0, end_idx);
                // Trim whitespace
                size_t start = function_name.find_first_not_of(" \t\n\r");
                if (start != std::string::npos) {
                    function_name = function_name.substr(start);
                }
                size_t end = function_name.find_last_not_of(" \t\n\r");
                if (end != std::string::npos) {
                    function_name = function_name.substr(0, end + 1);
                }

                params_region = fn_text.substr(end_idx + 1);
            }
        }

        if (function_name.empty()) {
            continue;
        }

        // Get parameter config for type coercion
        json param_config = find_tool_properties(tools, function_name);
        json args = json::object();

        // Extract parameters
        std::sregex_iterator param_iter(params_region.begin(), params_region.end(), *get_tool_call_parameter_re());
        std::sregex_iterator param_iter_end;
        for (std::sregex_iterator pit = param_iter; pit != param_iter_end; ++pit) {
            std::smatch pmatch = *pit;
            std::string match_text = pmatch[1].str();

            size_t eq_idx = match_text.find('>');
            if (eq_idx == std::string::npos) {
                continue;
            }

            std::string key = match_text.substr(0, eq_idx);
            std::string value = match_text.substr(eq_idx + 1);

            // Trim key
            size_t kstart = key.find_first_not_of(" \t\n\r");
            if (kstart != std::string::npos) {
                key = key.substr(kstart);
            }
            size_t kend = key.find_last_not_of(" \t\n\r");
            if (kend != std::string::npos) {
                key = key.substr(0, kend + 1);
            }

            // Trim value
            if (!value.empty() && value[0] == '\n') {
                value = value.substr(1);
            }
            if (!value.empty() && value.back() == '\n') {
                value = value.substr(0, value.size() - 1);
            }

            // Convert value to proper type
            std::string converted = convert_param_value(value, key, param_config, function_name);

            // Parse the converted value back into args
            try {
                json j = json::parse(converted);
                args[key] = j;
            } catch (...) {
                args[key] = value;
            }
        }

        // Create tool call
        ParsedToolCall tc;
        tc.id = generate_call_id();
        tc.type = "function";
        tc.function.name = function_name;
        tc.function.arguments = args.dump();

        tool_calls.push_back(std::move(tc));
    }

    // Add remaining content after last tool call
    cleaned_parts.push_back(text.substr(cursor));

    // Join cleaned parts
    std::ostringstream oss;
    for (const auto& part : cleaned_parts) {
        oss << part;
    }
    result.cleaned_content = oss.str();

    // Trim whitespace
    size_t start = result.cleaned_content.find_first_not_of(" \t\n\r");
    if (start != std::string::npos) {
        result.cleaned_content = result.cleaned_content.substr(start);
    }
    size_t end = result.cleaned_content.find_last_not_of(" \t\n\r");
    if (end != std::string::npos) {
        result.cleaned_content = result.cleaned_content.substr(0, end + 1);
    }

    result.tool_calls = std::move(tool_calls);
    return result;
}

// ─── Reasoning/Thinking Parsing ─────────────────────────────────────────────

ReasoningParseResult parse_reasoning(
    const std::string& text,
    bool thinking_enabled
) {
    ReasoningParseResult result;

    // Find </think> tag
    size_t think_open_pos = text.find(THINK_OPEN_TAG);

    std::string rest;
    if (think_open_pos != std::string::npos) {
        rest = text.substr(think_open_pos + strlen(THINK_OPEN_TAG));
    } else {
        rest = text;
    }

    // Find </think> tag
    size_t think_close_pos = rest.find(THINK_CLOSE_TAG);

    if (think_close_pos == std::string::npos) {
        if (thinking_enabled) {
            // No close tag - assume truncated; everything is reasoning
            result.content = "";
            result.reasoning = rest;
            // Trim whitespace
            size_t start = result.reasoning.find_first_not_of(" \t\n\r");
            if (start != std::string::npos) {
                result.reasoning = result.reasoning.substr(start);
            }
            if (!result.reasoning.empty()) {
                size_t end = result.reasoning.find_last_not_of(" \t\n\r");
                if (end != std::string::npos) {
                    result.reasoning = result.reasoning.substr(0, end + 1);
                }
            }
            if (result.reasoning.empty()) {
                result.reasoning.clear();
            }
        } else {
            // Thinking disabled - output is pure content
            result.content = rest;
            result.reasoning.clear();
        }
    } else {
        // Both tags present
        std::string reasoning = rest.substr(0, think_close_pos);
        std::string content = rest.substr(think_close_pos + strlen(THINK_CLOSE_TAG));

        // Trim whitespace
        size_t rstart = reasoning.find_first_not_of(" \t\n\r");
        if (rstart != std::string::npos) {
            reasoning = reasoning.substr(rstart);
        }
        if (!reasoning.empty()) {
            size_t rend = reasoning.find_last_not_of(" \t\n\r");
            if (rend != std::string::npos) {
                reasoning = reasoning.substr(0, rend + 1);
            }
        }

        size_t cstart = content.find_first_not_of(" \t\n\r");
        if (cstart != std::string::npos) {
            content = content.substr(cstart);
        }
        if (!content.empty()) {
            size_t cend = content.find_last_not_of(" \t\n\r");
            if (cend != std::string::npos) {
                content = content.substr(0, cend + 1);
            }
        }

        result.reasoning = reasoning.empty() ? std::string() : reasoning;
        result.content = content;
    }

    return result;
}

} // namespace dflash27b
