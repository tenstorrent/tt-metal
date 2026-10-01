// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/config.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>

#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <reflect>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/experimental/inspector_config.hpp>

namespace ttnn::core {

Config CONFIG{};

void Config::apply_json_overrides(const std::string& json_text, bool strict, const std::string& source) {
    const std::string from = source.empty() ? std::string{} : fmt::format(" (from {})", source);
    nlohmann::json json;
    try {
        json = nlohmann::json::parse(json_text);
    } catch (const nlohmann::json::exception& e) {
        // Wrapped: a bare nlohmann parse error does not say that TTNN config is what failed.
        TT_THROW("TTNN config overrides are not valid JSON: {}{}", e.what(), from);
    }
    TT_FATAL(json.is_object(), "TTNN config overrides must be a JSON object, got: {}{}", json_text, from);

    // Apply into a copy so a strict rejection leaves process-wide settings untouched.
    attributes_t next = this->attributes;
    std::vector<std::string_view> applied;
    for (const auto& [key, value] : json.items()) {
        bool known = false;
        reflect::for_each(
            [&](auto I) {
                auto name = reflect::member_name<I>(next);
                if (name != key) {
                    return;
                }
                known = true;
                auto& member = reflect::get<I>(next);
                using T = std::decay_t<decltype(member)>;
                try {
                    if constexpr (std::is_same_v<T, std::filesystem::path>) {
                        member = T{value.get<std::string>()};
                    } else if constexpr (std::is_same_v<T, std::optional<std::filesystem::path>>) {
                        member = value.is_null() ? T{} : T{value.get<std::string>()};
                    } else {
                        static_assert(
                            std::is_arithmetic_v<T> || std::is_same_v<T, std::string>,
                            "Config attribute type needs a JSON conversion in apply_json_overrides");
                        member = value.get<T>();
                    }
                } catch (const nlohmann::json::exception& e) {
                    // Strict callers get nothing applied; a config file keeps the keys that did parse.
                    if (strict) {
                        TT_THROW("Bad value for configuration key {}: {}{}", key, e.what(), from);
                    }
                    log_warning(tt::LogAlways, "Ignoring configuration key {}: {}{}", key, e.what(), from);
                    return;
                }
                applied.push_back(name);
            },
            next);
        if (!known) {
            if (strict) {
                TT_THROW("Unknown configuration key: {}{}", key, from);
            }
            log_warning(tt::LogAlways, "Unknown configuration key: {}{}", key, from);
        }
    }

    this->attributes = std::move(next);
    for (std::string_view name : applied) {
        this->validate(name);
    }
}

std::vector<std::string> Config::keys() {
    std::vector<std::string> names;
    reflect::for_each<attributes_t>([&](auto I) { names.emplace_back(reflect::member_name<I, attributes_t>()); });
    return names;
}

void Config::load_from_file(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        log_warning(tt::LogAlways, "Failed to load ttnn configuration from {}: cannot open the file", path.string());
        return;
    }
    std::stringstream text;
    text << file.rdbuf();
    try {
        this->apply_json_overrides(
            text.str(),
            /*strict=*/false,
            fmt::format("{}; update or delete it to get the new default config", path.string()));
    } catch (const std::exception& e) {
        // A stale or corrupt file must not fail the process, and this also runs at library load.
        log_warning(tt::LogAlways, "Failed to load ttnn configuration from {}: {}", path.string(), e.what());
    }
}

void Config::save_to_file(const std::filesystem::path& path) const {
    nlohmann::ordered_json json;
    reflect::for_each(
        [&](auto I) {
            const auto& member = reflect::get<I>(this->attributes);
            using T = std::decay_t<decltype(member)>;
            auto name = std::string(reflect::member_name<I>(this->attributes));
            if constexpr (std::is_same_v<T, std::filesystem::path>) {
                json[name] = member.string();
            } else if constexpr (std::is_same_v<T, std::optional<std::filesystem::path>>) {
                json[name] = member.has_value() ? nlohmann::ordered_json(member->string()) : nlohmann::ordered_json();
            } else {
                json[name] = member;
            }
        },
        this->attributes);
    std::ofstream file(path);
    TT_FATAL(file.is_open(), "Failed to open {} to save the TTNN configuration", path.string());
    file << json.dump(4);
}

// Apply both config sources at load so pure C++ consumers (e.g. the sanity-pipeline gtests) honor them
// too, in the order ttnn/__init__.py used to: the file first, then the env var on top.
static const int apply_env_config = [] {
    try {
        if (const char* config_path = std::getenv("TTNN_CONFIG_PATH")) {
            const std::filesystem::path path{config_path};
            if (std::filesystem::exists(path)) {
                CONFIG.load_from_file(path);
            } else {
                if (path.has_parent_path()) {
                    std::filesystem::create_directories(path.parent_path());
                }
                Config{}.save_to_file(path);
            }
        }
        if (const char* overrides = std::getenv("TTNN_CONFIG_OVERRIDES")) {
            CONFIG.apply_json_overrides(overrides, /*strict=*/true, "TTNN_CONFIG_OVERRIDES");
        }
    } catch (const std::exception& e) {
        // Nothing can catch this early (dlopen, pre-main), so fail here with the reason.
        // _Exit, not exit: running atexit handlers here deadlocks against the loader lock.
        log_critical(tt::LogAlways, "Cannot apply the TTNN configuration: {}", e.what());
        std::fflush(nullptr);
        std::_Exit(1);
    }
    return 0;
}();

std::vector<std::pair<std::string, std::string>> Config::get_config_entries() const {
    std::vector<std::pair<std::string, std::string>> entries;
    reflect::for_each(
        [&](auto I) {
            entries.emplace_back(
                std::string(reflect::member_name<I>(this->attributes)),
                fmt::format("{}", reflect::get<I>(this->attributes)));
        },
        this->attributes);
    return entries;
}

// Auto-register TTNN config with Inspector at library load time.
// The callback is invoked lazily when getConfiguration RPC is queried.
static const int register_inspector_config = [] {
    tt::tt_metal::inspector::add_config_callback([]() {
        const auto config_entries = CONFIG.get_config_entries();
        std::vector<tt::tt_metal::inspector::ConfigurationEntry> entries;
        entries.reserve(config_entries.size());
        for (const auto& [name, value] : config_entries) {
            entries.push_back(
                {name, value == "nullopt" ? "(unset)" : value, tt::tt_metal::inspector::ConfigScope::TtnnConfig});
        }
        return entries;
    });
    return 0;
}();

std::optional<std::filesystem::path> Config::get_report_path_impl() const {
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

            name_str.erase(
                std::unique(name_str.begin(), name_str.end(), [](char a, char b) { return a == '_' && b == '_'; }),
                name_str.end());

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
            date_time << month << std::setw(2) << std::setfill('0') << tm.tm_mday << "_" << std::setw(2)
                      << std::setfill('0') << tm.tm_hour << std::setw(2) << std::setfill('0') << tm.tm_min;

            std::string dir_name = name_str + "_" + date_time.str();
            cached_report_path = this->attributes.root_report_path / dir_name;
        }

        // snake_cased(report_name)_monthdd_HHMM
        return cached_report_path;
    }

    return std::nullopt;
}

void Config::validate(std::string_view name) const {
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

std::ostream& operator<<(std::ostream& os, const Config& config) {
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

}  // namespace ttnn::core

auto fmt::formatter<ttnn::Config>::format(const ttnn::Config& config, format_context& ctx) const
    -> format_context::iterator {
    std::stringstream ss;
    ss << config;
    return fmt::format_to(ctx.out(), "{}", ss.str());
}
