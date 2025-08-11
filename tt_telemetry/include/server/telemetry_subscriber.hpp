#pragma once

/*
 * telemetry_subscriber.hpp
 *
 * Interface for a telemetry consumer that accepts snapshots of telemetry data via HandoffHandle.
 */

 #include <server/telemetry_snapshot.hpp>
 #include <server/handoff_handle.hpp>

class TelemetrySubscriber {
public:
    virtual ~TelemetrySubscriber() = default;
    virtual void on_telemetry_ready(HandoffHandle<TelemetrySnapshot> &&telemetry) = 0;
 };