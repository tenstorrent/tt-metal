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

    // Physical link information (immutable, sent once per metric)
    // Maps metric path to physical link topology info as JSON
    std::unordered_map<std::string, nlohmann::json> physical_link_info;

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
        physical_link_info.clear();
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
    // Helper template to merge maps efficiently
    template <typename MapType>
    static void merge_map_into(MapType& target, const MapType& source) {
        for (const auto& [key, value] : source) {
            target[key] = value;
        }
    }

    /**
     * Fast path: merge without validation (original implementation)
     */
    void merge_from_fast(const TelemetrySnapshot& other) {
        // Update unit label maps if they are provided
        if (!other.metric_unit_display_label_by_code.empty() || !other.metric_unit_full_label_by_code.empty()) {
            merge_map_into(metric_unit_display_label_by_code, other.metric_unit_display_label_by_code);
            merge_map_into(metric_unit_full_label_by_code, other.metric_unit_full_label_by_code);
        }

        // Update metrics and their metadata using helper
        merge_map_into(bool_metrics, other.bool_metrics);
        merge_map_into(bool_metric_timestamps, other.bool_metric_timestamps);
        merge_map_into(uint_metrics, other.uint_metrics);
        merge_map_into(uint_metric_units, other.uint_metric_units);
        merge_map_into(uint_metric_timestamps, other.uint_metric_timestamps);
        merge_map_into(double_metrics, other.double_metrics);
        merge_map_into(double_metric_units, other.double_metric_units);
        merge_map_into(double_metric_timestamps, other.double_metric_timestamps);
        merge_map_into(string_metrics, other.string_metrics);
        merge_map_into(string_metric_units, other.string_metric_units);
        merge_map_into(string_metric_timestamps, other.string_metric_timestamps);

        // Merge physical link info (immutable, sent once per metric)
        for (const auto& [path, info] : other.physical_link_info) {
            physical_link_info.insert({path, info});
        }
    }

    // Helper template to validate and merge unit label maps
    template <typename T>
    void validate_and_merge_label_map(
        std::unordered_map<T, std::string>& target,
        const std::unordered_map<T, std::string>& source,
        const char* label_type) {
        for (const auto& [code, label] : source) {
            auto result = target.insert({code, label});
            if (!result.second && result.first->second != label) {
                log_error(
                    tt::LogAlways,
                    "Unit {} label redefinition detected for code {}: existing='{}', new='{}'",
                    label_type,
                    code,
                    result.first->second,
                    label);
            }
        }
    }

    /**
     * Validation path: merge with comprehensive validation checks
     */
    void merge_from_with_validation(const TelemetrySnapshot& other) {
        // Validate and merge unit label maps
        if (!other.metric_unit_display_label_by_code.empty() || !other.metric_unit_full_label_by_code.empty()) {
            validate_and_merge_label_map(
                metric_unit_display_label_by_code, other.metric_unit_display_label_by_code, "display");
            validate_and_merge_label_map(metric_unit_full_label_by_code, other.metric_unit_full_label_by_code, "full");
        }

        // Helper to check for metric type conflicts
        auto check_metric_conflict = [](const std::string& path,
                                        const char* new_type,
                                        const char* existing_type,
                                        const auto& existing_map) -> bool {
            if (existing_map.find(path) != existing_map.end()) {
                log_error(
                    tt::LogAlways,
                    "Metric type conflict: path '{}' exists as both {} and {}",
                    path,
                    new_type,
                    existing_type);
                return true;
            }
            return false;
        };

        // Validate and merge bool metrics
        for (const auto& [path, value] : other.bool_metrics) {
            if (check_metric_conflict(path, "bool", "uint", uint_metrics) ||
                check_metric_conflict(path, "bool", "double", double_metrics) ||
                check_metric_conflict(path, "bool", "string", string_metrics)) {
                continue;
            }
            bool_metrics[path] = value;
        }
        merge_map_into(bool_metric_timestamps, other.bool_metric_timestamps);

        // Validate and merge uint metrics
        for (const auto& [path, value] : other.uint_metrics) {
            if (check_metric_conflict(path, "uint", "bool", bool_metrics) ||
                check_metric_conflict(path, "uint", "double", double_metrics) ||
                check_metric_conflict(path, "uint", "string", string_metrics)) {
                continue;
            }
            uint_metrics[path] = value;
        }
        merge_map_into(uint_metric_units, other.uint_metric_units);
        merge_map_into(uint_metric_timestamps, other.uint_metric_timestamps);

        // Validate and merge double metrics
        for (const auto& [path, value] : other.double_metrics) {
            if (check_metric_conflict(path, "double", "bool", bool_metrics) ||
                check_metric_conflict(path, "double", "uint", uint_metrics) ||
                check_metric_conflict(path, "double", "string", string_metrics)) {
                continue;
            }
            double_metrics[path] = value;
        }
        merge_map_into(double_metric_units, other.double_metric_units);
        merge_map_into(double_metric_timestamps, other.double_metric_timestamps);

        // Validate and merge string metrics
        for (const auto& [path, value] : other.string_metrics) {
            if (check_metric_conflict(path, "string", "bool", bool_metrics) ||
                check_metric_conflict(path, "string", "uint", uint_metrics) ||
                check_metric_conflict(path, "string", "double", double_metrics)) {
                continue;
            }
            string_metrics[path] = value;
        }
        merge_map_into(string_metric_units, other.string_metric_units);
        merge_map_into(string_metric_timestamps, other.string_metric_timestamps);

        // Merge physical link info (immutable, sent once per metric)
        for (const auto& [path, info] : other.physical_link_info) {
            auto result = physical_link_info.insert({path, info});
            if (!result.second) {  // Key already exists
                if (result.first->second != info) {
                    log_error(
                        tt::LogAlways,
                        "Physical link info redefinition detected for path '{}': existing and new values differ",
                        path);
                }
            }
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
         {"physical_link_info", t.physical_link_info},
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

    // Physical link info may not be present in older snapshots
    if (j.contains("physical_link_info")) {
        j.at("physical_link_info").get_to(t.physical_link_info);
    }
}
