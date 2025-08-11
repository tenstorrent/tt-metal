#pragma once

/*
 * telemetry_snapshot.hpp
 *
 * Snapshot of telemetry data. Supports both deltas and absolute snapshots.
 */

 #include <string>
 #include <vector>

 struct TelemetrySnapshot {
    std::vector<size_t> metric_indices;
    std::vector<std::string> metric_names;
    std::vector<bool> metric_values;
    bool is_absolute;

    void clear() {
        metric_indices.clear();
        metric_names.clear();
        metric_values.clear();
    }
 };