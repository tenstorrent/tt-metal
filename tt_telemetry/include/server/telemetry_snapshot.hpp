#pragma once

/*
 * telemetry_snapshot.hpp
 *
 * Snapshot of telemetry data. Supports both deltas and absolute snapshots.
 */

 #include <string>
 #include <vector>

 #include <nlohmann/json.hpp>

 struct TelemetrySnapshot {
    std::vector<size_t> metric_indices;
    std::vector<std::string> metric_names;
    std::vector<uint8_t> metric_values;
    bool is_absolute;

    void clear() {
        metric_indices.clear();
        metric_names.clear();
        metric_values.clear();
    }
 };

 static inline void to_json(nlohmann::json &j, const TelemetrySnapshot &t) {
    j = nlohmann::json {
        { "metric_indices", t.metric_indices },
        { "metric_names", t.metric_names }, 
        { "metric_values", t.metric_values },
        { "is_absolute", t.is_absolute }
    };
}

static inline void from_json(const nlohmann::json &j, TelemetrySnapshot &t) {
    j.at("metric_indices").get_to(t.metric_indices);
    j.at("metric_names").get_to(t.metric_names);
    j.at("metric_values").get_to(t.metric_values);
    j.at("is_absolute").get_to(t.is_absolute);
}