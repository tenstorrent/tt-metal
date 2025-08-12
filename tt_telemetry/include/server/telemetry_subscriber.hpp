#pragma once

/*
 * server/telemetry_subscriber.hpp
 *
 * Interface for a telemetry consumer that accepts snapshots of telemetry data.
 */

 #include <server/telemetry_snapshot.hpp>

class TelemetrySubscriber {
public:
    virtual ~TelemetrySubscriber() = default;
    virtual void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) = 0;
 };