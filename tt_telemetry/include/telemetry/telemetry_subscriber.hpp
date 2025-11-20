#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/telemetry_subscriber.hpp
 *
 * Base class for a telemetry consumer that accepts snapshots of telemetry data.
 * Handles the common pattern of:
 *
 * - Receiving telemetry snapshots via on_telemetry_ready()
 * - Queueing them for processing
 * - Running a background thread that merges snapshots into accumulated state
 * - Calling on_telemetry_updated() for derived classes to handle the updated state
 */

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include <telemetry/telemetry_snapshot.hpp>
#include <utils/simple_concurrent_queue.hpp>

class TelemetrySubscriber {
public:
    TelemetrySubscriber();
    virtual ~TelemetrySubscriber();

    // Called by the telemetry collector when new telemetry data is ready.
    // This is thread-safe and simply queues the snapshot for processing.
    void on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry);

protected:
    // Called after telemetry has been merged into the state.
    // Derived classes should implement this to handle the updated telemetry.
    // @param delta The delta snapshot that was just merged
    //
    // Note: Derived classes can access telemetry_state_ if needed, but must lock state_mutex_ appropriately.
    virtual void on_telemetry_updated(const TelemetrySnapshot& delta) = 0;

    // Accumulated telemetry data (merged from all snapshots)
    TelemetrySnapshot telemetry_state_;

    // Mutex to protect access to telemetry_state_
    // Derived classes must use this when accessing telemetry_state_ from any thread
    std::mutex state_mutex_;

private:
    // Main telemetry processing loop
    void process_telemetry_loop();

    // Thread-safe queue of pending snapshots to process
    SimpleConcurrentQueue<std::shared_ptr<TelemetrySnapshot>> pending_snapshots_;

    // Background processing thread
    std::thread processing_thread_;
    std::atomic<bool> running_{false};
};
