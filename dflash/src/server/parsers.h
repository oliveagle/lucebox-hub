// parsers.h - Tool call and reasoning parsers for DFlash server
//
// Implements parsers for Qwen3 tool calling format and thinking/reasoning mode.
// Ported from vLLM's Qwen3CoderToolParser and Qwen3ReasoningParser.

#pragma once

#include <cstdint>
#include <regex>
#include <string>
#include <vector>
#include <optional>

#include "nlohmann/json.hpp"

namespace dflash27b {

// Forward declarations
struct ToolDef;

// ─── Tool Call Parsing ─────────────────────────────────────────────────────

// Parsed tool call in OpenAI format
struct ParsedToolCall {
    std::string id;
    std::string type = "function";
    struct {
        std::string name;
        std::string arguments;  // JSON string
    } function;
};

// Result of parsing tool calls from text
struct ToolCallParseResult {
    std::string cleaned_content;  // Text with tool call XML removed
    std::vector<ParsedToolCall> tool_calls;
};

// Parse Qwen3 tool calls from model output.
// Handles format: <invoke><function=NAME><parameter=KEY>VALUE</parameter>...</function></invoke>
//
// Args:
//   text: Raw model output
//   tools: Optional list of tool definitions for type coercion
//
// Returns:
//   ToolCallParseResult with cleaned content and parsed tool calls
ToolCallParseResult parse_tool_calls(
    const std::string& text,
    const std::vector<ToolDef>* tools = nullptr
);

// ─── Reasoning/Thinking Parsing ─────────────────────────────────────────────

// Result of parsing reasoning content
struct ReasoningParseResult {
    std::string content;       // Main response content
    std::string reasoning;     // Thinking/reasoning content (may be empty)
};

// Parse Qwen3 thinking/reasoning from model output.
// Handles format: <think>...</think>content
//
// The template typically prefills <think>\n into the prompt, so only
// </think> appears in generated output (headless mode). Older templates
// may emit both tags.
//
// Args:
//   text: Raw model output
//   thinking_enabled: Whether thinking mode is enabled (default: true)
//
// Returns:
//   ReasoningParseResult with separated content and reasoning
ReasoningParseResult parse_reasoning(
    const std::string& text,
    bool thinking_enabled = true
);

// ─── Helper Functions ───────────────────────────────────────────────────────

// Convert parameter value string to proper JSON type based on schema.
// Supports: int, float, bool, string, object, array, null
std::string convert_param_value(
    const std::string& param_value,
    const std::string& param_name,
    const nlohmann::json& param_config,
    const std::string& function_name
);

// Find tool properties by function name from tools list
nlohmann::json find_tool_properties(
    const std::vector<ToolDef>* tools,
    const std::string& function_name
);

} // namespace dflash27b
