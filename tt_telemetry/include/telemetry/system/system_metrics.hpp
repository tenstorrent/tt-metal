#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/system/system_metrics.hpp
 *
 * System-level telemetry metrics that track host-level health and status.
 * These metrics are independent of device telemetry.
 */

#include <memory>
#include <vector>

#include <telemetry/metric.hpp>

/*
 * TelemetryRunningMetric
 *
 * Boolean metric indicating whether telemetry collection is successfully running.
 * - true: Telemetry initialization succeeded and metrics are being collected
 * - false: Telemetry initialization failed (e.g., UMD initialization error)
 *
 * Path: system/TelemetryRunning
 */
class TelemetryRunningMetric : public BoolMetric {
public:
    TelemetryRunningMetric();

    const std::vector<std::string> telemetry_path() const override;

    // No update() needed - this metric is set manually via set_value()
};
