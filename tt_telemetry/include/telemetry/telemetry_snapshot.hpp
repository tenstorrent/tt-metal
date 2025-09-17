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

struct TelemetrySnapshot {
    // String-based telemetry data maps: path -> value
    std::unordered_map<std::string, bool> bool_metrics;
    std::unordered_map<std::string, uint64_t> uint_metrics;
    std::unordered_map<std::string, double> double_metrics;

    // Metadata maps: path -> metadata
    std::unordered_map<std::string, uint64_t> bool_metric_timestamps;
    std::unordered_map<std::string, uint16_t> uint_metric_units;
    std::unordered_map<std::string, uint64_t> uint_metric_timestamps;
    std::unordered_map<std::string, uint16_t> double_metric_units;
    std::unordered_map<std::string, uint64_t> double_metric_timestamps;

    // Unit label maps
    std::unordered_map<uint16_t, std::string> metric_unit_display_label_by_code;
    std::unordered_map<uint16_t, std::string> metric_unit_full_label_by_code;

    void clear() {
        bool_metrics.clear();
        uint_metrics.clear();
        double_metrics.clear();
        bool_metric_timestamps.clear();
        uint_metric_units.clear();
        uint_metric_timestamps.clear();
        double_metric_units.clear();
        double_metric_timestamps.clear();
        metric_unit_display_label_by_code.clear();
        metric_unit_full_label_by_code.clear();
    }

    /**
     * Merge another snapshot into this one.
     * This replaces the functionality of TelemetryDataStore::update_from_snapshot().
     * @param other The snapshot to merge into this one
     */
    void merge_from(const TelemetrySnapshot& other) {
        // Update unit label maps if they are provided
        if (!other.metric_unit_display_label_by_code.empty() || !other.metric_unit_full_label_by_code.empty()) {
            // Merge unit label maps (TODO: assert we don't have any redefinitions)
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
    }
};

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
     j = nlohmann::json{
         {"bool_metrics", t.bool_metrics},
         {"uint_metrics", t.uint_metrics},
         {"double_metrics", t.double_metrics},
         {"bool_metric_timestamps", t.bool_metric_timestamps},
         {"uint_metric_units", t.uint_metric_units},
         {"uint_metric_timestamps", t.uint_metric_timestamps},
         {"double_metric_units", t.double_metric_units},
         {"double_metric_timestamps", t.double_metric_timestamps},
         {"metric_unit_display_label_by_code", t.metric_unit_display_label_by_code},
         {"metric_unit_full_label_by_code", t.metric_unit_full_label_by_code},
     };
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("bool_metrics").get_to(t.bool_metrics);
    j.at("uint_metrics").get_to(t.uint_metrics);
    j.at("double_metrics").get_to(t.double_metrics);
    j.at("bool_metric_timestamps").get_to(t.bool_metric_timestamps);
    j.at("uint_metric_units").get_to(t.uint_metric_units);
    j.at("uint_metric_timestamps").get_to(t.uint_metric_timestamps);
    j.at("double_metric_units").get_to(t.double_metric_units);
    j.at("double_metric_timestamps").get_to(t.double_metric_timestamps);
    j.at("metric_unit_display_label_by_code").get_to(t.metric_unit_display_label_by_code);
    j.at("metric_unit_full_label_by_code").get_to(t.metric_unit_full_label_by_code);
}
