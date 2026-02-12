// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>

#include <fmt/format.h>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/stl_fmt.hpp>

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

    // Named getters
    auto get_cache_path() const { return attributes.cache_path; }
    auto get_model_cache_path() const { return attributes.model_cache_path; }
    auto get_tmp_dir() const { return attributes.tmp_dir; }
    auto get_enable_model_cache() const { return attributes.enable_model_cache; }
    auto get_enable_fast_runtime_mode() const { return attributes.enable_fast_runtime_mode; }
    auto get_throw_exception_on_fallback() const { return attributes.throw_exception_on_fallback; }
    auto get_enable_logging() const { return attributes.enable_logging; }
    auto get_enable_graph_report() const { return attributes.enable_graph_report; }
    auto get_enable_detailed_buffer_report() const { return attributes.enable_detailed_buffer_report; }
    auto get_enable_detailed_tensor_report() const { return attributes.enable_detailed_tensor_report; }
    auto get_enable_comparison_mode() const { return attributes.enable_comparison_mode; }
    auto get_comparison_mode_should_raise_exception() const {
        return attributes.comparison_mode_should_raise_exception;
    }
    auto get_comparison_mode_pcc() const { return attributes.comparison_mode_pcc; }
    auto get_root_report_path() const { return attributes.root_report_path; }
    auto get_report_name() const { return attributes.report_name; }

    // Named setters
    void set_cache_path(const std::filesystem::path& v) {
        attributes.cache_path = v;
        validate("cache_path");
    }
    void set_model_cache_path(const std::filesystem::path& v) {
        attributes.model_cache_path = v;
        validate("model_cache_path");
    }
    void set_tmp_dir(const std::filesystem::path& v) {
        attributes.tmp_dir = v;
        validate("tmp_dir");
    }
    void set_enable_model_cache(bool v) {
        attributes.enable_model_cache = v;
        validate("enable_model_cache");
    }
    void set_enable_fast_runtime_mode(bool v) {
        attributes.enable_fast_runtime_mode = v;
        validate("enable_fast_runtime_mode");
    }
    void set_throw_exception_on_fallback(bool v) {
        attributes.throw_exception_on_fallback = v;
        validate("throw_exception_on_fallback");
    }
    void set_enable_logging(bool v) {
        attributes.enable_logging = v;
        validate("enable_logging");
    }
    void set_enable_graph_report(bool v) {
        attributes.enable_graph_report = v;
        validate("enable_graph_report");
    }
    void set_enable_detailed_buffer_report(bool v) {
        attributes.enable_detailed_buffer_report = v;
        validate("enable_detailed_buffer_report");
    }
    void set_enable_detailed_tensor_report(bool v) {
        attributes.enable_detailed_tensor_report = v;
        validate("enable_detailed_tensor_report");
    }
    void set_enable_comparison_mode(bool v) {
        attributes.enable_comparison_mode = v;
        validate("enable_comparison_mode");
    }
    void set_comparison_mode_should_raise_exception(bool v) {
        attributes.comparison_mode_should_raise_exception = v;
        validate("comparison_mode_should_raise_exception");
    }
    void set_comparison_mode_pcc(float v) {
        attributes.comparison_mode_pcc = v;
        validate("comparison_mode_pcc");
    }
    void set_root_report_path(const std::filesystem::path& v) {
        attributes.root_report_path = v;
        validate("root_report_path");
    }
    void set_report_name(const std::optional<std::filesystem::path>& v) {
        attributes.report_name = v;
        validate("report_name");
    }

    // report_path getter (computed property)
    std::optional<std::filesystem::path> get_report_path() const {
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
                    if (std::isalnum(c)) {
                        return static_cast<char>(std::tolower(c));
                    }
                    return '_';
                });

                name_str.erase(std::unique(name_str.begin(), name_str.end(), [](char a, char b) {
                    return a == '_' && b == '_';
                }), name_str.end());

                if (!name_str.empty() && name_str.front() == '_') {
                    name_str.erase(0, 1);
                }
                if (!name_str.empty() && name_str.back() == '_') {
                    name_str.pop_back();
                }

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

    void validate(std::string_view name) const {
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
        os << "cache_path=" << fmt::format("{}", config.attributes.cache_path) << ",";
        os << "model_cache_path=" << fmt::format("{}", config.attributes.model_cache_path) << ",";
        os << "tmp_dir=" << fmt::format("{}", config.attributes.tmp_dir) << ",";
        os << "enable_model_cache=" << config.attributes.enable_model_cache << ",";
        os << "enable_fast_runtime_mode=" << config.attributes.enable_fast_runtime_mode << ",";
        os << "throw_exception_on_fallback=" << config.attributes.throw_exception_on_fallback << ",";
        os << "enable_logging=" << config.attributes.enable_logging << ",";
        os << "enable_graph_report=" << config.attributes.enable_graph_report << ",";
        os << "enable_detailed_buffer_report=" << config.attributes.enable_detailed_buffer_report << ",";
        os << "enable_detailed_tensor_report=" << config.attributes.enable_detailed_tensor_report << ",";
        os << "enable_comparison_mode=" << config.attributes.enable_comparison_mode << ",";
        os << "comparison_mode_should_raise_exception=" << config.attributes.comparison_mode_should_raise_exception
           << ",";
        os << "comparison_mode_pcc=" << config.attributes.comparison_mode_pcc << ",";
        os << "root_report_path=" << fmt::format("{}", config.attributes.root_report_path) << ",";
        os << "report_name=" << fmt::format("{}", config.attributes.report_name) << ",";
        os << "report_path=" << fmt::format("{}", config.get_report_path());
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
