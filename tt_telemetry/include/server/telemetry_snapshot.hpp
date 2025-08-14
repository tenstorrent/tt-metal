#pragma once

/*
 * server/telemetry_snapshot.hpp
 *
 * Snapshot of telemetry data. Supports both deltas and absolute snapshots. This is serialized
 * directly to JSON and sent to web clients.
 */

 #include <string>
 #include <vector>

 #include <nlohmann/json.hpp>

 struct TelemetrySnapshot {
    std::vector<size_t> bool_metric_ids;
    std::vector<std::string> bool_metric_names;
    std::vector<uint8_t> bool_metric_values;
    std::vector<size_t> int_metric_ids;
    std::vector<std::string> int_metric_names;
    std::vector<int> int_metric_values;
    bool is_absolute;

    void clear() {
        bool_metric_ids.clear();
        bool_metric_names.clear();
        bool_metric_values.clear();
        int_metric_ids.clear();
        int_metric_names.clear();
        int_metric_values.clear();
    }
 };

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
    j = nlohmann::json {
        { "bool_metric_ids", t.bool_metric_ids },
        { "bool_metric_names", t.bool_metric_names }, 
        { "bool_metric_values", t.bool_metric_values },
        { "int_metric_ids", t.int_metric_ids },
        { "int_metric_names", t.int_metric_names }, 
        { "int_metric_values", t.int_metric_values },
        { "is_absolute", t.is_absolute }
    };
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("bool_metric_ids").get_to(t.bool_metric_ids);
    j.at("bool_metric_names").get_to(t.bool_metric_names);
    j.at("bool_metric_values").get_to(t.bool_metric_values);
    j.at("int_metric_ids").get_to(t.int_metric_ids);
    j.at("int_metric_names").get_to(t.int_metric_names);
    j.at("int_metric_values").get_to(t.int_metric_values);
    j.at("is_absolute").get_to(t.is_absolute);
}