// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * System-level telemetry metrics that track host health and status.
 */

#include <telemetry/system/system_metrics.hpp>

/**************************************************************************************************
| TelemetryRunningMetric Class
**************************************************************************************************/

TelemetryRunningMetric::TelemetryRunningMetric() : BoolMetric() {
    value_ = false;  // Initially not running
}

const std::vector<std::string> TelemetryRunningMetric::telemetry_path() const { return {"system", "TelemetryRunning"}; }
