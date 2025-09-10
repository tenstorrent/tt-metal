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
 * Metric ID and value arrays must be the same length because the elements correspond to each
 * other. Names arrays are empty when updates to existing metrics are being transmitted.
 * Otherwise, the names array must be the same length as the ID and value arrays, and contains the
 * names of new metrics being transmitted for the first time.
 */

#include <string>
#include <vector>
#include <unordered_map>

#include <nlohmann/json.hpp>
#include <telemetry/metric.hpp>

struct TelemetrySnapshot {
    std::vector<size_t> bool_metric_ids;
    std::vector<std::string> bool_metric_names;
    std::vector<uint8_t> bool_metric_values;
    std::vector<uint64_t> bool_metric_timestamps;
    std::vector<size_t> uint_metric_ids;
    std::vector<std::string> uint_metric_names;
    std::vector<uint16_t> uint_metric_units;
    std::vector<uint64_t> uint_metric_values;
    std::vector<uint64_t> uint_metric_timestamps;
    std::vector<size_t> double_metric_ids;
    std::vector<std::string> double_metric_names;
    std::vector<uint16_t> double_metric_units;
    std::vector<double> double_metric_values;
    std::vector<uint64_t> double_metric_timestamps;

    // Unit label maps - populated when names are populated, empty otherwise
    std::unordered_map<uint16_t, std::string> metric_unit_display_label_by_code;
    std::unordered_map<uint16_t, std::string> metric_unit_full_label_by_code;

    void clear() {
        bool_metric_ids.clear();
        bool_metric_names.clear();
        bool_metric_values.clear();
        bool_metric_timestamps.clear();
        uint_metric_ids.clear();
        uint_metric_names.clear();
        uint_metric_units.clear();
        uint_metric_values.clear();
        uint_metric_timestamps.clear();
        double_metric_ids.clear();
        double_metric_names.clear();
        double_metric_units.clear();
        double_metric_values.clear();
        double_metric_timestamps.clear();
        metric_unit_display_label_by_code.clear();
        metric_unit_full_label_by_code.clear();
    }
};

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
     j = nlohmann::json{
         {"bool_metric_ids", t.bool_metric_ids},
         {"bool_metric_names", t.bool_metric_names},
         {"bool_metric_values", t.bool_metric_values},
         {"bool_metric_timestamps", t.bool_metric_timestamps},
         {"uint_metric_ids", t.uint_metric_ids},
         {"uint_metric_names", t.uint_metric_names},
         {"uint_metric_units", t.uint_metric_units},
         {"uint_metric_values", t.uint_metric_values},
         {"uint_metric_timestamps", t.uint_metric_timestamps},
         {"double_metric_ids", t.double_metric_ids},
         {"double_metric_names", t.double_metric_names},
         {"double_metric_units", t.double_metric_units},
         {"double_metric_values", t.double_metric_values},
         {"double_metric_timestamps", t.double_metric_timestamps},
         {"metric_unit_display_label_by_code", t.metric_unit_display_label_by_code},
         {"metric_unit_full_label_by_code", t.metric_unit_full_label_by_code},
     };
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("bool_metric_ids").get_to(t.bool_metric_ids);
    j.at("bool_metric_names").get_to(t.bool_metric_names);
    j.at("bool_metric_values").get_to(t.bool_metric_values);
    j.at("bool_metric_timestamps").get_to(t.bool_metric_timestamps);
    j.at("uint_metric_ids").get_to(t.uint_metric_ids);
    j.at("uint_metric_names").get_to(t.uint_metric_names);
    j.at("uint_metric_units").get_to(t.uint_metric_units);
    j.at("uint_metric_values").get_to(t.uint_metric_values);
    j.at("uint_metric_timestamps").get_to(t.uint_metric_timestamps);
    j.at("double_metric_ids").get_to(t.double_metric_ids);
    j.at("double_metric_names").get_to(t.double_metric_names);
    j.at("double_metric_units").get_to(t.double_metric_units);
    j.at("double_metric_values").get_to(t.double_metric_values);
    j.at("double_metric_timestamps").get_to(t.double_metric_timestamps);
    j.at("metric_unit_display_label_by_code").get_to(t.metric_unit_display_label_by_code);
    j.at("metric_unit_full_label_by_code").get_to(t.metric_unit_full_label_by_code);
}
