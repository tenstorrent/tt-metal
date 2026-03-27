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
#include <source_location>
#include <string>
#include <string_view>

#include <tt_stl/caseless_comparison.hpp>

namespace ttsl::gha {

// Usage example:
//   const auto ann = ttsl::gha::make_annotation(
//       ttsl::gha::annotation_level::warning, "Detected unexpected value");
//   ttsl::gha::emit_annotation(ann);
//
//   // Convenience form that captures file/line/function at the call site:
//   ttsl::gha::emit_annotation_at(
//       ttsl::gha::annotation_level::error, "Validation failed");

inline constexpr std::string_view github_actions_env_var = "GITHUB_ACTIONS";

enum class annotation_level {
    notice,
    warning,
    error,
};

// The string_view fields (message, file, title) do NOT own their data.
// The caller must ensure the referenced strings outlive the annotation object.
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

namespace detail {

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
    return env_var_is_truthy(std::getenv(github_actions_env_var.data()));
}

// Fills unset location fields from loc. Call-site capture must use a wrapper that defaults
// std::source_location::current() (see make_annotation / emit_annotation_at).
[[nodiscard]] constexpr annotation annotation_at(annotation ann, const std::source_location& loc) noexcept {
    if (ann.file.empty()) {
        ann.file = loc.file_name();
    }
    if (!ann.line.has_value()) {
        ann.line = static_cast<std::uint_least32_t>(loc.line());
    }
    if (!ann.column.has_value() && loc.column() != 0) {
        ann.column = static_cast<std::uint_least32_t>(loc.column());
    }
    if (ann.title.empty()) {
        ann.title = loc.function_name();
    }
    return ann;
}

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

}  // namespace detail

[[nodiscard]] constexpr annotation make_annotation(
    annotation_level level,
    std::string_view message,
    std::source_location loc = std::source_location::current()) noexcept {
    annotation ann{.level = level, .message = message};
    return detail::annotation_at(ann, loc);
}

[[nodiscard]] constexpr annotation make_annotation(
    const annotation& ann, std::source_location loc = std::source_location::current()) noexcept {
    return detail::annotation_at(ann, loc);
}

inline void emit_annotation(const annotation& annotation, std::FILE* stream = stdout) {
    // A leading newline helps GitHub workflow commands survive mpirun --tag-output:
    // the launcher may consume the empty tagged line, leaving the actual command
    // at column 0 where GitHub can parse it.
    fmt::print(stream, "\n{}\n", detail::build_workflow_command(annotation));
    std::fflush(stream);
}

inline void emit_annotation_at(
    annotation_level level,
    std::string_view message,
    std::FILE* stream = stdout,
    std::source_location loc = std::source_location::current()) {
    emit_annotation(make_annotation(level, message, loc), stream);
}

inline void emit_annotation_at(
    const annotation& ann, std::FILE* stream = stdout, std::source_location loc = std::source_location::current()) {
    emit_annotation(make_annotation(ann, loc), stream);
}

}  // namespace ttsl::gha
