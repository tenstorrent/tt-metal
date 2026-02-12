// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <iosfwd>
#include <optional>
#include <reflect>
#include <string>
#include <string_view>
#include <tuple>

#include <fmt/base.h>

namespace ttnn {

namespace core {

struct Config {
    struct attributes_t {
        std::filesystem::path cache_path = std::filesystem::path{std::getenv("HOME")} / ".cache/ttnn";
        std::filesystem::path model_cache_path = std::filesystem::path{std::getenv("HOME")} / ".cache/ttnn/models";
        std::filesystem::path tmp_dir = "/tmp/ttnn";
        bool enable_model_cache = false;
        bool enable_fast_runtime_mode = true;
        bool throw_exception_on_fallback = false;
        bool enable_logging = false;
        bool enable_graph_report = false;
        bool enable_detailed_buffer_report = false;
        bool enable_detailed_tensor_report = false;
        bool enable_comparison_mode = false;
        bool comparison_mode_should_raise_exception = false;
        float comparison_mode_pcc = 0.9999;
        std::filesystem::path root_report_path = "generated/ttnn/reports";
        std::optional<std::filesystem::path> report_name = std::nullopt;
    };

private:
    attributes_t attributes;
    mutable std::optional<std::filesystem::path> cached_report_path;
    mutable std::optional<std::string> cached_report_name;

    // Implementation helper for the report_path getter (defined in config.cpp).
    std::optional<std::filesystem::path> get_report_path_impl() const;

public:
    Config(auto&&... args) : attributes{std::forward<decltype(args)>(args)...} {}

    template <reflect::fixed_string name>
        requires requires { reflect::get<name>(std::declval<attributes_t>()); }
    auto get() const {
        return reflect::get<name>(this->attributes);
    }

    template <std::size_t index>
    auto get() const {
        return reflect::get<index>(this->attributes);
    }

    template <reflect::fixed_string name>
        requires(name == reflect::fixed_string{"report_path"})
    std::optional<std::filesystem::path> get() const {
        return get_report_path_impl();
    }

    template <
        reflect::fixed_string name,
        typename T = std::decay_t<decltype(reflect::get<name>(std::declval<attributes_t>()))>>
    void set(const T& value) {
        reflect::get<name>(this->attributes) = value;
        this->validate(name);
    }

    template <std::size_t index, typename T = std::decay_t<decltype(reflect::get<index>(std::declval<attributes_t>()))>>
    void set(const T& value) {
        reflect::get<index>(this->attributes) = value;
        this->validate(reflect::member_name<index>(this->attributes));
    }

    // Defined in config.cpp (uses tt-logger).
    void validate(std::string_view name) const;

    // Defined in config.cpp (uses reflection for_each).
    friend std::ostream& operator<<(std::ostream& os, const Config& config);
};

extern Config CONFIG;

}  // namespace core

using core::CONFIG;
using core::Config;
}  // namespace ttnn

template <>
struct fmt::formatter<ttnn::Config> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.end(); }

    // Defined in config.cpp.
    auto format(const ttnn::Config& config, format_context& ctx) const -> format_context::iterator;
};
