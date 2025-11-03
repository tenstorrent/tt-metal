#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/telemetry_snapshot.hpp
 *
 * Snapshot of telemetry data. Supports both deltas and absolute snapshots. This is serialized
 * directly to JSON and sent to web clients.
 *
 * Uses string-based telemetry paths as keys instead of integer IDs. Each metric type (bool, uint, double)
 * has its own unordered_map from path string to value. Metadata like timestamps and units are stored
 * in separate maps using the same path strings as keys.
 */

#include <string>
#include <vector>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <telemetry/metric.hpp>
#include <tt-logger/tt-logger.hpp>

struct TelemetrySnapshot {
    // String-based telemetry data maps: path -> value
    std::unordered_map<std::string, bool> bool_metrics;
    std::unordered_map<std::string, uint64_t> uint_metrics;
    std::unordered_map<std::string, double> double_metrics;
    std::unordered_map<std::string, std::string> string_metrics;

    // Metadata maps: path -> metadata
    std::unordered_map<std::string, uint64_t> bool_metric_timestamps;
    std::unordered_map<std::string, uint16_t> uint_metric_units;
    std::unordered_map<std::string, uint64_t> uint_metric_timestamps;
    std::unordered_map<std::string, uint16_t> double_metric_units;
    std::unordered_map<std::string, uint64_t> double_metric_timestamps;
    std::unordered_map<std::string, uint16_t> string_metric_units;
    std::unordered_map<std::string, uint64_t> string_metric_timestamps;

    // Unit label maps
    std::unordered_map<uint16_t, std::string> metric_unit_display_label_by_code;
    std::unordered_map<uint16_t, std::string> metric_unit_full_label_by_code;

    void clear() {
        bool_metrics.clear();
        uint_metrics.clear();
        double_metrics.clear();
        string_metrics.clear();
        bool_metric_timestamps.clear();
        uint_metric_units.clear();
        uint_metric_timestamps.clear();
        double_metric_units.clear();
        double_metric_timestamps.clear();
        string_metric_units.clear();
        string_metric_timestamps.clear();
        metric_unit_display_label_by_code.clear();
        metric_unit_full_label_by_code.clear();
    }

    /**
     * Merge another snapshot into this one.
     * @param other The snapshot to merge into this one
     * @param validate If true (default), perform validation checks for consistency
     */
    void merge_from(const TelemetrySnapshot& other, bool validate = true) {
        if (validate) {
            merge_from_with_validation(other);
        } else {
            merge_from_fast(other);
        }
    }

private:
    /**
     * Fast path: merge without validation (original implementation)
     */
    void merge_from_fast(const TelemetrySnapshot& other) {
        // Update unit label maps if they are provided
        if (!other.metric_unit_display_label_by_code.empty() || !other.metric_unit_full_label_by_code.empty()) {
            for (const auto& [code, label] : other.metric_unit_display_label_by_code) {
                metric_unit_display_label_by_code[code] = label;
            }
            for (const auto& [code, label] : other.metric_unit_full_label_by_code) {
                metric_unit_full_label_by_code[code] = label;
            }
        }

        // Update bool metrics and their metadata
        for (const auto& [path, value] : other.bool_metrics) {
            bool_metrics[path] = value;
        }
        for (const auto& [path, timestamp] : other.bool_metric_timestamps) {
            bool_metric_timestamps[path] = timestamp;
        }

        // Update uint metrics and their metadata
        for (const auto& [path, value] : other.uint_metrics) {
            uint_metrics[path] = value;
        }
        for (const auto& [path, unit] : other.uint_metric_units) {
            uint_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.uint_metric_timestamps) {
            uint_metric_timestamps[path] = timestamp;
        }

        // Update double metrics and their metadata
        for (const auto& [path, value] : other.double_metrics) {
            double_metrics[path] = value;
        }
        for (const auto& [path, unit] : other.double_metric_units) {
            double_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.double_metric_timestamps) {
            double_metric_timestamps[path] = timestamp;
        }

        // Update string metrics and their metadata
        for (const auto& [path, value] : other.string_metrics) {
            string_metrics[path] = value;
        }
        for (const auto& [path, unit] : other.string_metric_units) {
            string_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.string_metric_timestamps) {
            string_metric_timestamps[path] = timestamp;
        }
    }

    /**
     * Validation path: merge with comprehensive validation checks
     */
    void merge_from_with_validation(const TelemetrySnapshot& other) {
        // Validate and merge unit label maps
        if (!other.metric_unit_display_label_by_code.empty() || !other.metric_unit_full_label_by_code.empty()) {
            // Validate display label map
            for (const auto& [code, label] : other.metric_unit_display_label_by_code) {
                auto result = metric_unit_display_label_by_code.insert({code, label});
                if (!result.second) {  // Key already exists
                    if (result.first->second != label) {
                        log_error(
                            tt::LogAlways,
                            "Unit display label redefinition detected for code {}: existing='{}', new='{}'",
                            code,
                            result.first->second,
                            label);
                    }
                }
            }

            // Validate full label map
            for (const auto& [code, label] : other.metric_unit_full_label_by_code) {
                auto result = metric_unit_full_label_by_code.insert({code, label});
                if (!result.second) {  // Key already exists
                    if (result.first->second != label) {
                        log_error(
                            tt::LogAlways,
                            "Unit full label redefinition detected for code {}: existing='{}', new='{}'",
                            code,
                            result.first->second,
                            label);
                    }
                }
            }
        }

        // Validate and merge bool metrics
        for (const auto& [path, value] : other.bool_metrics) {
            // Check if this path exists in other metric types
            if (uint_metrics.find(path) != uint_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both bool and uint", path);
                continue;  // Skip this update
            }
            if (double_metrics.find(path) != double_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both bool and double", path);
                continue;  // Skip this update
            }
            bool_metrics[path] = value;
        }

        // Merge bool metadata
        for (const auto& [path, timestamp] : other.bool_metric_timestamps) {
            bool_metric_timestamps[path] = timestamp;
        }

        // Validate and merge uint metrics
        for (const auto& [path, value] : other.uint_metrics) {
            // Check if this path exists in other metric types
            if (bool_metrics.find(path) != bool_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both uint and bool", path);
                continue;  // Skip this update
            }
            if (double_metrics.find(path) != double_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both uint and double", path);
                continue;  // Skip this update
            }
            uint_metrics[path] = value;
        }

        // Merge uint metadata
        for (const auto& [path, unit] : other.uint_metric_units) {
            uint_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.uint_metric_timestamps) {
            uint_metric_timestamps[path] = timestamp;
        }

        // Validate and merge double metrics
        for (const auto& [path, value] : other.double_metrics) {
            // Check if this path exists in other metric types
            if (bool_metrics.find(path) != bool_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both double and bool", path);
                continue;  // Skip this update
            }
            if (uint_metrics.find(path) != uint_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both double and uint", path);
                continue;  // Skip this update
            }
            double_metrics[path] = value;
        }

        // Merge double metadata
        for (const auto& [path, unit] : other.double_metric_units) {
            double_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.double_metric_timestamps) {
            double_metric_timestamps[path] = timestamp;
        }

        // Validate and merge string metrics
        for (const auto& [path, value] : other.string_metrics) {
            // Check if this path exists in other metric types
            if (bool_metrics.find(path) != bool_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both string and bool", path);
                continue;  // Skip this update
            }
            if (uint_metrics.find(path) != uint_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both string and uint", path);
                continue;  // Skip this update
            }
            if (double_metrics.find(path) != double_metrics.end()) {
                log_error(tt::LogAlways, "Metric type conflict: path '{}' exists as both string and double", path);
                continue;  // Skip this update
            }
            string_metrics[path] = value;
        }

        // Merge string metadata
        for (const auto& [path, unit] : other.string_metric_units) {
            string_metric_units[path] = unit;
        }
        for (const auto& [path, timestamp] : other.string_metric_timestamps) {
            string_metric_timestamps[path] = timestamp;
        }
    }

public:
};

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
     j = nlohmann::json{
         {"bool_metrics", t.bool_metrics},
         {"uint_metrics", t.uint_metrics},
         {"double_metrics", t.double_metrics},
         {"string_metrics", t.string_metrics},
         {"bool_metric_timestamps", t.bool_metric_timestamps},
         {"uint_metric_units", t.uint_metric_units},
         {"uint_metric_timestamps", t.uint_metric_timestamps},
         {"double_metric_units", t.double_metric_units},
         {"double_metric_timestamps", t.double_metric_timestamps},
         {"string_metric_units", t.string_metric_units},
         {"string_metric_timestamps", t.string_metric_timestamps},
         {"metric_unit_display_label_by_code", t.metric_unit_display_label_by_code},
         {"metric_unit_full_label_by_code", t.metric_unit_full_label_by_code},
     };
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("bool_metrics").get_to(t.bool_metrics);
    j.at("uint_metrics").get_to(t.uint_metrics);
    j.at("double_metrics").get_to(t.double_metrics);
    j.at("string_metrics").get_to(t.string_metrics);
    j.at("bool_metric_timestamps").get_to(t.bool_metric_timestamps);
    j.at("uint_metric_units").get_to(t.uint_metric_units);
    j.at("uint_metric_timestamps").get_to(t.uint_metric_timestamps);
    j.at("double_metric_units").get_to(t.double_metric_units);
    j.at("double_metric_timestamps").get_to(t.double_metric_timestamps);
    j.at("string_metric_units").get_to(t.string_metric_units);
    j.at("string_metric_timestamps").get_to(t.string_metric_timestamps);
    j.at("metric_unit_display_label_by_code").get_to(t.metric_unit_display_label_by_code);
    j.at("metric_unit_full_label_by_code").get_to(t.metric_unit_full_label_by_code);
}
