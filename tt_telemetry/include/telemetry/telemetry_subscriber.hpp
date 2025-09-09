#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/telemetry_subscriber.hpp
 *
 * Interface for a telemetry consumer that accepts snapshots of telemetry data.
 */

 #include <telemetry/telemetry_snapshot.hpp>

class TelemetrySubscriber {
public:
    virtual ~TelemetrySubscriber() = default;
    virtual void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) = 0;
};
