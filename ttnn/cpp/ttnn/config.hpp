// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <reflect>
#include <string>
#include <string_view>
#include <tuple>

#include "tt_metal/common/logger.hpp"
#include "tt_metal/tt_stl/reflection.hpp"

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

public:
    Config(auto&&... args) : attributes{std::forward<decltype(args)>(args)...} {}

    template <reflect::fixed_string name>
        requires requires { reflect::get<name>(std::declval<attributes_t>()); }
    const auto get() const {
        return reflect::get<name>(this->attributes);
    }

    template <std::size_t index>
    const auto get() const {
        return reflect::get<index>(this->attributes);
    }

    template <reflect::fixed_string name>
        requires(name == reflect::fixed_string{"report_path"})
    const std::optional<std::filesystem::path> get() const {
        if (this->attributes.report_name.has_value()) {
            auto hash = std::hash<std::string>{}(this->attributes.report_name.value());
            return this->attributes.root_report_path / std::to_string(hash);
        }
        return std::nullopt;
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

    void validate(std::string_view name) {
        if (name == "enable_fast_runtime_mode" or name == "enable_logging") {
            if (this->attributes.enable_fast_runtime_mode) {
                if (this->attributes.enable_logging) {
                    tt::log_warning(
                        tt::LogAlways,
                        "Logging cannot be enabled in fast runtime mode. Please disable fast runtime mode if you want "
                        "to enable logging.");
                }
            }
        }

        if (name == "enable_comparison_mode") {
            if (this->attributes.enable_fast_runtime_mode && this->attributes.enable_comparison_mode) {
                tt::log_warning(
                    tt::LogAlways,
                    "Comparison mode is currently not supported with fast runtime mode enabled. Please disable fast "
                    "runtime mode ('enable_fast_runtime_mode = false') to use tensor comparison mode.");
            }
        }

        if (name == "enable_fast_runtime_mode" or name == "enable_graph_report" or
            name == "enable_detailed_buffer_report" or name == "enable_detailed_tensor_report") {
            if (not this->attributes.enable_logging) {
                if (this->attributes.enable_graph_report) {
                    tt::log_warning(
                        tt::LogAlways, "Running without logging. Please enable logging to save graph report");
                }
                if (this->attributes.enable_detailed_buffer_report) {
                    tt::log_warning(
                        tt::LogAlways, "Running without logging. Please enable logging to save detailed buffer report");
                }
                if (this->attributes.enable_detailed_tensor_report) {
                    tt::log_warning(
                        tt::LogAlways, "Running without logging. Please enable logging to save detailed tensor report");
                }
            }
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Config& config) {
        os << "Config{";
        reflect::for_each(
            [&](auto I) {
                os << reflect::member_name<I>(config.attributes) << "="
                   << fmt::format("{}", reflect::get<I>(config.attributes)) << ",";
            },
            config.attributes);
        os << fmt::format("{}", config.get<"report_path">());
        os << "}";
        return os;
    }
};

extern Config CONFIG;

}  // namespace core

using core::CONFIG;
using core::Config;
}  // namespace ttnn

template <>
struct fmt::formatter<ttnn::Config> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.end(); }

    auto format(const ttnn::Config& config, format_context& ctx) const -> format_context::iterator {
        std::stringstream ss;
        ss << config;
        return fmt::format_to(ctx.out(), "{}", ss.str());
    }
};
