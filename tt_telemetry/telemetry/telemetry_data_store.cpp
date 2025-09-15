// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/telemetry_data_store.hpp>
#include <tt-metalium/assert.hpp>

TelemetrySnapshot TelemetryDataStore::create_full_snapshot() const {
    std::lock_guard<std::mutex> lock(data_mutex_);

    TelemetrySnapshot full_snapshot;

    // Populate bool metrics
    for (const auto& [id, name] : bool_metric_name_by_id_) {
        full_snapshot.bool_metric_ids.push_back(id);
        full_snapshot.bool_metric_names.push_back(name);
        full_snapshot.bool_metric_values.push_back(bool_metric_value_by_id_.at(id));
    }

    // Populate uint metrics
    for (const auto& [id, name] : uint_metric_name_by_id_) {
        full_snapshot.uint_metric_ids.push_back(id);
        full_snapshot.uint_metric_names.push_back(name);
        full_snapshot.uint_metric_units.push_back(uint_metric_units_by_id_.at(id));
        full_snapshot.uint_metric_values.push_back(uint_metric_value_by_id_.at(id));
    }

    // Populate double metrics
    for (const auto& [id, name] : double_metric_name_by_id_) {
        full_snapshot.double_metric_ids.push_back(id);
        full_snapshot.double_metric_names.push_back(name);
        full_snapshot.double_metric_units.push_back(double_metric_units_by_id_.at(id));
        full_snapshot.double_metric_values.push_back(double_metric_value_by_id_.at(id));
    }

    // Include cached unit label maps
    full_snapshot.metric_unit_display_label_by_code = metric_unit_display_label_by_code_;
    full_snapshot.metric_unit_full_label_by_code = metric_unit_full_label_by_code_;

    return full_snapshot;
}

void TelemetryDataStore::update_from_snapshot(const TelemetrySnapshot& snapshot) {
    std::lock_guard<std::mutex> lock(data_mutex_);

    // Validate snapshot consistency (same logic as original)
    TT_ASSERT(snapshot.bool_metric_ids.size() == snapshot.bool_metric_values.size());
    if (snapshot.bool_metric_names.size() > 0) {
        TT_ASSERT(snapshot.bool_metric_ids.size() == snapshot.bool_metric_names.size());
    }
    TT_ASSERT(snapshot.uint_metric_ids.size() == snapshot.uint_metric_values.size());
    if (snapshot.uint_metric_names.size() > 0) {
        TT_ASSERT(snapshot.uint_metric_ids.size() == snapshot.uint_metric_names.size());
        TT_ASSERT(snapshot.uint_metric_ids.size() == snapshot.uint_metric_units.size());
    }
    TT_ASSERT(snapshot.double_metric_ids.size() == snapshot.double_metric_values.size());
    if (snapshot.double_metric_names.size() > 0) {
        TT_ASSERT(snapshot.double_metric_ids.size() == snapshot.double_metric_names.size());
        TT_ASSERT(snapshot.double_metric_ids.size() == snapshot.double_metric_units.size());
    }

    // Cache unit label maps when any names are populated
    if (snapshot.uint_metric_names.size() > 0 || snapshot.double_metric_names.size() > 0) {
        metric_unit_display_label_by_code_ = snapshot.metric_unit_display_label_by_code;
        metric_unit_full_label_by_code_ = snapshot.metric_unit_full_label_by_code;
    }

    // Update bool metrics
    for (size_t i = 0; i < snapshot.bool_metric_ids.size(); i++) {
        size_t idx = snapshot.bool_metric_ids[i];
        if (snapshot.bool_metric_names.size() > 0) {
            // Names were included, which indicates new metrics added!
            bool_metric_name_by_id_[idx] = snapshot.bool_metric_names[i];
        }
        bool_metric_value_by_id_[idx] = snapshot.bool_metric_values[i];
    }

    // Update uint metrics
    for (size_t i = 0; i < snapshot.uint_metric_ids.size(); i++) {
        size_t idx = snapshot.uint_metric_ids[i];
        if (snapshot.uint_metric_names.size() > 0) {
            // Names were included, which indicates new metrics added!
            uint_metric_name_by_id_[idx] = snapshot.uint_metric_names[i];
            uint_metric_units_by_id_[idx] = snapshot.uint_metric_units[i];
        }
        uint_metric_value_by_id_[idx] = snapshot.uint_metric_values[i];
    }

    // Update double metrics
    for (size_t i = 0; i < snapshot.double_metric_ids.size(); i++) {
        size_t idx = snapshot.double_metric_ids[i];
        if (snapshot.double_metric_names.size() > 0) {
            // Names were included, which indicates new metrics added!
            double_metric_name_by_id_[idx] = snapshot.double_metric_names[i];
            double_metric_units_by_id_[idx] = snapshot.double_metric_units[i];
        }
        double_metric_value_by_id_[idx] = snapshot.double_metric_values[i];
    }
}
