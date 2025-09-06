#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/arc/arc_metrics.hpp
 *
 * ARC telemetry metrics that read from ARC (Argonaut RISC Core) processors on chips.
 * Includes both integral (uint) and floating-point (double) valued metrics.
 */

#include <memory>
#include <string>

#include <telemetry/metric.hpp>
#include <telemetry/arc/arc_telemetry_reader.hpp>


class ARCTelemetryAvailableMetric : public BoolMetric {
public:
    // Constructor
    ARCTelemetryAvailableMetric(size_t chip_id, std::shared_ptr<ARCTelemetryReader> reader);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
};

class ARCUintMetric : public UIntMetric {
public:
    // Constructor for direct telemetry tag usage
    ARCUintMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        MetricUnit units = MetricUnit::UNITLESS);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::TelemetryTag tag_;
    std::string metric_name_;
    uint32_t mask_;
};

class ARCDoubleMetric : public DoubleMetric {
public:
    // Enum to specify whether telemetry values are signed or unsigned
    enum class Signedness {
        UNSIGNED,
        SIGNED
    };

    // Constructor for direct telemetry tag usage
    ARCDoubleMetric(
        size_t chip_id,
        std::shared_ptr<ARCTelemetryReader> reader,
        tt::umd::TelemetryTag tag,
        const std::string& metric_name,
        uint32_t mask = 0xffffffff,
        double scale_factor = 1.0,
        MetricUnit units = MetricUnit::UNITLESS,
        Signedness signedness = Signedness::UNSIGNED);

    const std::vector<std::string> telemetry_path() const override;
    void update(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle) override;

private:
    std::shared_ptr<ARCTelemetryReader> reader_;
    tt::umd::TelemetryTag tag_;
    std::string metric_name_;
    uint32_t mask_;        // Mask to apply to raw telemetry value
    double scale_factor_;  // Factor to scale raw telemetry value to double
    Signedness signedness_;  // Whether the value is signed or unsigned
};
