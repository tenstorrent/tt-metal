#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/telemetry_collector.hpp
 *
 * Polls telemetry data on a periodic loop and sends to subscribers.
 */

#include <string>
#include <vector>

#include <telemetry/telemetry_subscriber.hpp>

namespace tt::llrt {
class RunTimeOptions;
}

namespace tt::scaleout_tools::fsd::proto {
class FactorySystemDescriptor;
}

void run_telemetry_collector(
    bool telemetry_enabled,
    std::vector<std::shared_ptr<TelemetrySubscriber>> subscribers,
    const std::vector<std::string>& aggregate_endpoints,
    const tt::llrt::RunTimeOptions& rtoptions,
    tt::scaleout_tools::fsd::proto::FactorySystemDescriptor fsd,
    int watchdog_timeout_seconds,
    int failure_exposure_duration_seconds,
    bool mmio_only = false);
