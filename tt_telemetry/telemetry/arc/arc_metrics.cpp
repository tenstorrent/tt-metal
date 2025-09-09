// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * ARC telemetry is described in the ISA documentation:
 * https://github.com/tenstorrent/tt-isa-documentation/blob/main/WormholeB0/ARCTile/Telemetry.md
 */

#include <tt-metalium/assert.hpp>
 #include <telemetry/arc/arc_metrics.hpp>

#include <chrono>

/**************************************************************************************************
| ARCTelemetryAvailableMetric Class
**************************************************************************************************/

ARCTelemetryAvailableMetric::ARCTelemetryAvailableMetric(
    size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader) :
    BoolMetric(chip_id, MetricUnit::UNITLESS),
    reader_(reader) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    value_ = false;
}

const std::vector<std::string> ARCTelemetryAvailableMetric::telemetry_path() const {
    // Start with the chip identifier path
    std::vector<std::string> path = reader_->id.telemetry_path();

    // Add the metric name
    path.push_back("ARCTelemetryAvailable");

    return path;
}

void ARCTelemetryAvailableMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    bool new_value = reader_->is_valid();

    // Update the metric value and timestamp
    bool old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}

/**************************************************************************************************
| ARCUintMetric Class
**************************************************************************************************/

ARCUintMetric::ARCUintMetric(
    size_t chip_id,
    std::shared_ptr<ARCTelemetryReader> reader,
    tt::umd::TelemetryTag tag,
    const std::string& metric_name,
    uint32_t mask,
    MetricUnit units) :
    UIntMetric(chip_id, units),
    reader_(reader),
    tag_(tag),
    metric_name_(metric_name),
    mask_(mask) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    value_ = 0;
}

const std::vector<std::string> ARCUintMetric::telemetry_path() const {
    // Start with the chip identifier path
    std::vector<std::string> path = reader_->id.telemetry_path();

    // Add the metric name
    path.push_back(metric_name_);

    return path;
}

void ARCUintMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Don't attempt to read if telemetry reader is invalid
    if (!reader_->is_valid()) {
        return;
    }

    // Read the telemetry value using common telemetry tag
    uint32_t raw_value = reader_->read_value(tag_);

    // Apply mask to get the final value
    uint64_t new_value = static_cast<uint64_t>(raw_value & mask_);

    // Update the metric value and timestamp
    uint64_t old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}

/**************************************************************************************************
| ARCDoubleMetric Class
**************************************************************************************************/

ARCDoubleMetric::ARCDoubleMetric(
    size_t chip_id,
    std::shared_ptr<ARCTelemetryReader> reader,
    tt::umd::TelemetryTag tag,
    const std::string& metric_name,
    uint32_t mask,
    double scale_factor,
    MetricUnit units,
    Signedness signedness) :
    DoubleMetric(chip_id, units),
    reader_(reader),
    tag_(tag),
    metric_name_(metric_name),
    mask_(mask),
    scale_factor_(scale_factor),
    signedness_(signedness) {
    TT_ASSERT(reader_ != nullptr, "ARCTelemetryReader cannot be null");
    TT_FATAL(
        signedness == Signedness::UNSIGNED || signedness == Signedness::SIGNED,
        "Signedness must be either UNSIGNED or SIGNED");
    value_ = 0.0;
}

const std::vector<std::string> ARCDoubleMetric::telemetry_path() const {
    // Start with the chip identifier path
    std::vector<std::string> path = reader_->id.telemetry_path();

    // Add the metric name
    path.push_back(metric_name_);

    return path;
}

void ARCDoubleMetric::update(
    const std::unique_ptr<tt::umd::Cluster>& cluster, std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Don't attempt to read if telemetry reader is invalid
    if (!reader_->is_valid()) {
        return;
    }

    // Read the telemetry value using common telemetry tag
    uint32_t raw_value = reader_->read_value(tag_);

    // Apply mask to get the final raw value
    uint32_t masked_value = raw_value & mask_;

    // Convert to double and apply scale factor
    double new_value;
    if (signedness_ == Signedness::SIGNED) {
        // For signed values, cast to int32_t first to handle sign extension
        int32_t signed_value = static_cast<int32_t>(masked_value);
        new_value = static_cast<double>(signed_value) * scale_factor_;
    } else {
        // For unsigned values, cast directly to double
        new_value = static_cast<double>(masked_value) * scale_factor_;
    }

    // Update the metric value and timestamp
    double old_value = value_;
    changed_since_transmission_ = new_value != old_value;
    value_ = new_value;
    timestamp_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
}
