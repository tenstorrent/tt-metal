#pragma once

/*
 * telemetry/telemetry_snapshot.hpp
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
    std::vector<size_t> uint_metric_ids;
    std::vector<std::string> uint_metric_names;
    std::vector<uint64_t> uint_metric_values;
    bool is_absolute;

    void clear() {
        bool_metric_ids.clear();
        bool_metric_names.clear();
        bool_metric_values.clear();
        uint_metric_ids.clear();
        uint_metric_names.clear();
        uint_metric_values.clear();
    }
 };

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
     j = nlohmann::json{
         {"bool_metric_ids", t.bool_metric_ids},
         {"bool_metric_names", t.bool_metric_names},
         {"bool_metric_values", t.bool_metric_values},
         {"uint_metric_ids", t.uint_metric_ids},
         {"uint_metric_names", t.uint_metric_names},
         {"uint_metric_values", t.uint_metric_values},
         {"is_absolute", t.is_absolute}};
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("bool_metric_ids").get_to(t.bool_metric_ids);
    j.at("bool_metric_names").get_to(t.bool_metric_names);
    j.at("bool_metric_values").get_to(t.bool_metric_values);
    j.at("uint_metric_ids").get_to(t.uint_metric_ids);
    j.at("uint_metric_names").get_to(t.uint_metric_names);
    j.at("uint_metric_values").get_to(t.uint_metric_values);
    j.at("is_absolute").get_to(t.is_absolute);
}
