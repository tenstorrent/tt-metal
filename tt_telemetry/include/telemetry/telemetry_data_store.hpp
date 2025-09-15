#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/telemetry_data_store.hpp
 *
 * Simple data store for telemetry metrics. Stores telemetry data in unordered maps
 * and provides methods to populate and update from TelemetrySnapshot instances.
 */

#include <unordered_map>
#include <string>
#include <memory>
#include <mutex>

#include <telemetry/telemetry_snapshot.hpp>

class TelemetryDataStore {
private:
    // Telemetry data - same structure as in original TelemetryServer
    std::unordered_map<size_t, std::string> bool_metric_name_by_id_;
    std::unordered_map<size_t, bool> bool_metric_value_by_id_;
    std::unordered_map<size_t, std::string> uint_metric_name_by_id_;
    std::unordered_map<size_t, uint16_t> uint_metric_units_by_id_;
    std::unordered_map<size_t, uint64_t> uint_metric_value_by_id_;
    std::unordered_map<size_t, std::string> double_metric_name_by_id_;
    std::unordered_map<size_t, uint16_t> double_metric_units_by_id_;
    std::unordered_map<size_t, double> double_metric_value_by_id_;
    std::unordered_map<uint16_t, std::string> metric_unit_display_label_by_code_;
    std::unordered_map<uint16_t, std::string> metric_unit_full_label_by_code_;

    mutable std::mutex data_mutex_;

public:
    TelemetryDataStore() = default;
    ~TelemetryDataStore() = default;

    // Non-copyable, non-movable
    TelemetryDataStore(const TelemetryDataStore&) = delete;
    TelemetryDataStore& operator=(const TelemetryDataStore&) = delete;
    TelemetryDataStore(TelemetryDataStore&&) = delete;
    TelemetryDataStore& operator=(TelemetryDataStore&&) = delete;

    /**
     * Create a full telemetry snapshot from current data store state.
     * @return TelemetrySnapshot containing all current telemetry data
     */
    TelemetrySnapshot create_full_snapshot() const;

    /**
     * Update data store from a telemetry snapshot.
     * Uses the same logic as the original update_telemetry_state_from_snapshot().
     * @param snapshot Const reference to TelemetrySnapshot to update from
     */
    void update_from_snapshot(const TelemetrySnapshot& snapshot);
};
