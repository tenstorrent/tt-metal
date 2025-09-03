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

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/reflection.hpp>

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
        if (this->attributes.report_name.has_value()) {
            std::string name_str = this->attributes.report_name.value().string();

            // Only recompute if report_name has changed in this run
            if (!cached_report_name.has_value() || *cached_report_name != name_str) {
                cached_report_name = name_str;

                // If report_name is too long, truncate it
                constexpr size_t max_name_length = 64;
                if (name_str.length() > max_name_length) {
                    name_str = name_str.substr(0, max_name_length);
                }

                std::transform(name_str.begin(), name_str.end(), name_str.begin(), [](unsigned char c) {
                    if (std::isalnum(c)) return static_cast<char>(std::tolower(c));
                    return '_';
                });

                name_str.erase(std::unique(name_str.begin(), name_str.end(), [](char a, char b) {
                    return a == '_' && b == '_';
                }), name_str.end());

                if (!name_str.empty() && name_str.front() == '_') name_str.erase(0, 1);
                if (!name_str.empty() && name_str.back() == '_') name_str.pop_back();

                // Get current date and time
                auto now = std::chrono::system_clock::now();
                std::time_t now_c = std::chrono::system_clock::to_time_t(now);
                std::tm tm{};
            #if defined(_WIN32)
                localtime_s(&tm, &now_c);
            #else
                localtime_r(&now_c, &tm);
            #endif
                std::ostringstream oss;
                oss << std::put_time(&tm, "%b");
                std::string month = oss.str();
                std::transform(month.begin(), month.end(), month.begin(), ::tolower);

                std::ostringstream date_time;
                date_time << month
                        << std::setw(2) << std::setfill('0') << tm.tm_mday << "_"
                        << std::setw(2) << std::setfill('0') << tm.tm_hour
                        << std::setw(2) << std::setfill('0') << tm.tm_min;

                std::string dir_name = name_str + "_" + date_time.str();
                cached_report_path = this->attributes.root_report_path / dir_name;
            }

            // snake_cased(report_name)_monthdd_HHMM
            return cached_report_path;
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
                    log_warning(
                        tt::LogAlways,
                        "Logging cannot be enabled in fast runtime mode. Please disable fast runtime mode if you want "
                        "to enable logging.");
                }
            }
        }

        if (name == "enable_comparison_mode") {
            if (this->attributes.enable_fast_runtime_mode && this->attributes.enable_comparison_mode) {
                log_warning(
                    tt::LogAlways,
                    "Comparison mode is currently not supported with fast runtime mode enabled. Please disable fast "
                    "runtime mode ('enable_fast_runtime_mode = false') to use tensor comparison mode.");
            }
        }

        if (name == "enable_fast_runtime_mode" or name == "enable_graph_report" or
            name == "enable_detailed_buffer_report" or name == "enable_detailed_tensor_report") {
            if (not this->attributes.enable_logging) {
                if (this->attributes.enable_graph_report) {
                    log_warning(tt::LogAlways, "Running without logging. Please enable logging to save graph report");
                }
                if (this->attributes.enable_detailed_buffer_report) {
                    log_warning(
                        tt::LogAlways, "Running without logging. Please enable logging to save detailed buffer report");
                }
                if (this->attributes.enable_detailed_tensor_report) {
                    log_warning(
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
