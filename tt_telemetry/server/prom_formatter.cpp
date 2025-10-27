// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <server/prom_formatter.hpp>
#include <tt-logger/tt-logger.hpp>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <ctime>

namespace tt::telemetry {

namespace {

// Structure to hold parsed metric information
struct ParsedMetric {
    std::string metric_name;
    std::unordered_map<std::string, std::string> labels;
};

// Helper to extract hostname from path and return remaining path
// Path format: hostname/rest/of/path
// Returns: {hostname, rest/of/path}
std::pair<std::string, std::string_view> extract_hostname_from_path(std::string_view path) {
    size_t first_slash = path.find('/');
    if (first_slash == std::string_view::npos) {
        throw std::runtime_error("Metric path does not contain hostname separator: " + std::string(path));
    }

    std::string hostname(path.substr(0, first_slash));
    std::string_view remaining = path.substr(first_slash + 1);
    return {hostname, remaining};
}

// Parse system metric path (simpler format: hostname/system/MetricName)
// Returns: metric_name with hostname as a label (labels from metric itself merged later)
ParsedMetric parse_system_metric_path(std::string_view path) {
    ParsedMetric result;
    auto [hostname, path_after_hostname] = extract_hostname_from_path(path);

    // Add hostname as a label
    result.labels["hostname"] = hostname;

    // Expected format: system/MetricName
    constexpr std::string_view system_prefix = "system/";
    if (path_after_hostname.starts_with(system_prefix)) {
        result.metric_name = std::string(path_after_hostname.substr(system_prefix.length()));
    } else {
        throw std::runtime_error("System metric path must start with 'system/': " + std::string(path_after_hostname));
    }

    return result;
}

// Parse metric path and extract labels
// Path format: hostname/tray3/chip0/[channel5/]metric_name
// Returns: metric_name and labels (hostname, tray, chip, channel if present)
ParsedMetric parse_metric_path(std::string_view path) {
    ParsedMetric result;
    auto [hostname, path_after_hostname] = extract_hostname_from_path(path);

    // Add hostname as a label
    result.labels["hostname"] = hostname;

    // Split remaining path into components
    std::vector<std::string> components;
    std::string path_str(path_after_hostname);
    size_t start = 0;
    size_t end = path_str.find('/');
    while (end != std::string::npos) {
        components.push_back(path_str.substr(start, end - start));
        start = end + 1;
        end = path_str.find('/', start);
    }
    components.push_back(path_str.substr(start));  // Last component

    if (components.empty()) {
        throw std::runtime_error("Empty metric path after removing hostname: " + std::string(path));
    }

    // Last component is always the metric name
    result.metric_name = components.back();

    // Parse hierarchical labels (tray, chip, channel, etc.)
    for (size_t i = 0; i < components.size() - 1; ++i) {
        const std::string& component = components[i];

        // Extract label name and value (e.g., "tray3" -> label="tray", value="3")
        auto digit_it =
            std::find_if(component.begin(), component.end(), [](unsigned char c) { return std::isdigit(c); });

        if (digit_it == component.begin() || digit_it == component.end()) {
            // No label/value structure, skip or use as-is
            continue;
        }

        std::string label_name = component.substr(0, std::distance(component.begin(), digit_it));
        std::string label_value = component.substr(std::distance(component.begin(), digit_it));
        result.labels[label_name] = label_value;
    }

    return result;
}

// Helper to format a single metric in Prometheus format with labels
void format_metric(
    std::stringstream& output,
    std::string_view metric_name,
    const std::unordered_map<std::string, std::string>& labels,
    std::string_view value,
    std::string_view help_text,
    std::string_view unit_label,
    uint64_t timestamp,
    std::unordered_set<std::string>& written_metric_names) {
    // Write HELP and TYPE only once per unique metric name
    if (written_metric_names.insert(std::string(metric_name)).second) {
        output << "# HELP " << metric_name << " " << help_text << "\n";
        output << "# TYPE " << metric_name << " gauge\n";
    }

    // Write metric line with labels
    output << metric_name << "{";

    // Add extracted labels (tray, chip, channel, etc.)
    bool first_label = true;
    for (const auto& [label_name, label_value] : labels) {
        if (!first_label) {
            output << ",";
        }
        output << label_name << "=\"" << label_value << "\"";
        first_label = false;
    }

    // Add unit label if present
    if (!unit_label.empty()) {
        if (!first_label) {
            output << ",";
        }
        output << "unit=\"" << unit_label << "\"";
    }

    output << "} " << value;

    if (timestamp > 0) {
        output << " " << timestamp;
    }
    output << "\n\n";
}

// Template helper to process metrics with common logic
template <typename ValueType, typename ValueConverter>
void process_metrics(
    std::stringstream& output,
    const std::unordered_map<std::string, ValueType>& metrics,
    const std::unordered_map<std::string, uint64_t>& timestamps,
    const std::unordered_map<std::string, uint16_t>* units,
    const std::unordered_map<uint16_t, std::string>& unit_labels,
    std::string_view help_text,
    ValueConverter value_converter) {
    std::unordered_set<std::string> written_metric_names;

    for (const auto& [path, value] : metrics) {
        try {
            // Parse metric path to extract name and labels (including hostname)
            ParsedMetric parsed = parse_metric_path(path);
            std::string value_str = value_converter(value);

            // Get timestamp
            uint64_t timestamp = 0;
            auto ts_it = timestamps.find(path);
            if (ts_it != timestamps.end()) {
                timestamp = ts_it->second;
            }

            // Get unit label (if units map provided)
            std::string unit_label;
            if (units != nullptr) {
                auto unit_it = units->find(path);
                if (unit_it != units->end()) {
                    auto label_it = unit_labels.find(unit_it->second);
                    if (label_it != unit_labels.end()) {
                        unit_label = label_it->second;
                    }
                }
            }

            format_metric(
                output,
                parsed.metric_name,
                parsed.labels,
                value_str,
                help_text,
                unit_label,
                timestamp,
                written_metric_names);
        } catch (const std::exception& e) {
            // Log warning but continue processing other metrics
            log_warning(tt::LogAlways, "Failed to format metric '{}': {}", path, e.what());
        }
    }
}

// Template helper to process system-level metrics
template <typename ValueType, typename ValueConverter>
void process_system_metrics(
    std::stringstream& output,
    const std::unordered_map<std::string, ValueType>& metrics,
    const std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& labels_map,
    const std::unordered_map<std::string, uint64_t>& timestamps,
    std::string_view help_text,
    ValueConverter value_converter) {
    std::unordered_set<std::string> written_metric_names;

    for (const auto& [path, value] : metrics) {
        try {
            // Parse system metric path (hostname/system/MetricName)
            // Hostname is extracted from path and added as a label
            ParsedMetric parsed = parse_system_metric_path(path);
            std::string value_str = value_converter(value);

            // Get timestamp
            uint64_t timestamp = 0;
            auto ts_it = timestamps.find(path);
            if (ts_it != timestamps.end()) {
                timestamp = ts_it->second;
            }

            // Merge labels from labels map (these override path-extracted labels if conflicts)
            auto labels_it = labels_map.find(path);
            if (labels_it != labels_map.end()) {
                for (const auto& [label_name, label_value] : labels_it->second) {
                    parsed.labels[label_name] = label_value;
                }
            }

            format_metric(
                output, parsed.metric_name, parsed.labels, value_str, help_text, "", timestamp, written_metric_names);
        } catch (const std::exception& e) {
            // Log warning but continue processing other metrics
            log_warning(tt::LogAlways, "Failed to format system metric '{}': {}", path, e.what());
        }
    }
}

}  // anonymous namespace

std::string format_snapshot_as_prometheus(const TelemetrySnapshot& snapshot) {
    std::stringstream output;

    // Write header
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&now_c, &tm_buf);

    output << "# Tenstorrent Metal Telemetry Metrics\n";
    output << "# Generated at: " << std::put_time(&tm_buf, "%c") << "\n\n";

    // Process system metrics first (host-level health)
    process_system_metrics(
        output,
        snapshot.system_bool_metrics,
        snapshot.system_bool_metric_labels,
        snapshot.system_bool_metric_timestamps,
        "System-level health metric",
        [](bool v) { return std::to_string(v ? 1 : 0); });

    // Process bool metrics (no units)
    process_metrics(
        output,
        snapshot.bool_metrics,
        snapshot.bool_metric_timestamps,
        nullptr,  // No units for bool metrics
        snapshot.metric_unit_display_label_by_code,
        "Boolean metric from Tenstorrent Metal",
        [](bool v) { return std::to_string(v ? 1 : 0); });

    // Process unsigned integer metrics (with units)
    process_metrics(
        output,
        snapshot.uint_metrics,
        snapshot.uint_metric_timestamps,
        &snapshot.uint_metric_units,
        snapshot.metric_unit_display_label_by_code,
        "Unsigned integer metric from Tenstorrent Metal",
        [](uint64_t v) { return std::to_string(v); });

    // Process floating-point metrics (with units)
    process_metrics(
        output,
        snapshot.double_metrics,
        snapshot.double_metric_timestamps,
        &snapshot.double_metric_units,
        snapshot.metric_unit_display_label_by_code,
        "Floating-point metric from Tenstorrent Metal",
        [](double v) { return std::to_string(v); });

    return output.str();
}

}  // namespace tt::telemetry
