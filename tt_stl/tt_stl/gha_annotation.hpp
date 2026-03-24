// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fmt/format.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>

#include <tt_stl/caseless_comparison.hpp>

namespace ttsl::gha {

inline constexpr char github_actions_env_var[] = "GITHUB_ACTIONS";

enum class annotation_level {
    notice,
    warning,
    error,
};

struct annotation {
    annotation_level level{annotation_level::notice};
    std::string_view message{};
    std::string_view file{};
    std::optional<std::uint_least32_t> line{};
    std::optional<std::uint_least32_t> end_line{};
    std::optional<std::uint_least32_t> column{};
    std::optional<std::uint_least32_t> end_column{};
    std::string_view title{};
};

[[nodiscard]] constexpr std::string_view to_string(annotation_level level) noexcept {
    switch (level) {
        case annotation_level::notice: return "notice";
        case annotation_level::warning: return "warning";
        case annotation_level::error: return "error";
    }
    return "notice";
}

[[nodiscard]] constexpr bool env_value_is_truthy(std::string_view value) noexcept {
    return value == "1" || ascii_caseless_comp(value, std::string_view("true")) ||
           ascii_caseless_comp(value, std::string_view("yes")) || ascii_caseless_comp(value, std::string_view("on"));
}

[[nodiscard]] inline bool env_var_is_truthy(const char* value) noexcept {
    return value != nullptr && env_value_is_truthy(value);
}

[[nodiscard]] inline bool should_emit_annotations(const char* explicit_enable_env_var = nullptr) noexcept {
    if (explicit_enable_env_var != nullptr) {
        if (const char* value = std::getenv(explicit_enable_env_var); value != nullptr) {
            return env_var_is_truthy(value);
        }
    }
    return env_var_is_truthy(std::getenv(github_actions_env_var));
}

namespace detail {

inline void append_escaped_text(fmt::memory_buffer& out, std::string_view text, bool property_context) {
    auto append_literal = [&](std::string_view literal) { out.append(literal.begin(), literal.end()); };
    for (char ch : text) {
        switch (ch) {
            case '%': append_literal("%25"); break;
            case '\r': append_literal("%0D"); break;
            case '\n': append_literal("%0A"); break;
            case ':':
                if (property_context) {
                    append_literal("%3A");
                } else {
                    out.push_back(ch);
                }
                break;
            case ',':
                if (property_context) {
                    append_literal("%2C");
                } else {
                    out.push_back(ch);
                }
                break;
            default: out.push_back(ch); break;
        }
    }
}

}  // namespace detail

[[nodiscard]] inline std::string build_workflow_command(const annotation& annotation) {
    fmt::memory_buffer command;
    fmt::format_to(std::back_inserter(command), "::{}", to_string(annotation.level));

    bool wrote_property = false;
    auto maybe_write_property_prefix = [&]() {
        if (!wrote_property) {
            command.push_back(' ');
            wrote_property = true;
        } else {
            command.push_back(',');
        }
    };
    auto append_property = [&](std::string_view key, std::string_view value) {
        maybe_write_property_prefix();
        command.append(key.begin(), key.end());
        command.push_back('=');
        detail::append_escaped_text(command, value, true);
    };
    auto append_numeric_property = [&](std::string_view key, std::optional<std::uint_least32_t> value) {
        if (!value.has_value()) {
            return;
        }
        maybe_write_property_prefix();
        fmt::format_to(std::back_inserter(command), "{}={}", key, *value);
    };

    if (!annotation.file.empty()) {
        append_property("file", annotation.file);
    }
    append_numeric_property("line", annotation.line);
    append_numeric_property("endLine", annotation.end_line);
    append_numeric_property("col", annotation.column);
    append_numeric_property("endColumn", annotation.end_column);
    if (!annotation.title.empty()) {
        append_property("title", annotation.title);
    }

    command.push_back(':');
    command.push_back(':');
    detail::append_escaped_text(command, annotation.message, false);
    return fmt::to_string(command);
}

inline void emit_annotation(const annotation& annotation, std::FILE* stream = stdout) {
    fmt::print(stream, "{}\n", build_workflow_command(annotation));
    std::fflush(stream);
}

}  // namespace ttsl::gha
