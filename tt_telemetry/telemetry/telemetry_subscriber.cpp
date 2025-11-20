// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/telemetry_subscriber.hpp>
#include <tt-logger/tt-logger.hpp>

TelemetrySubscriber::TelemetrySubscriber() {
    // Start the background processing thread
    running_ = true;
    processing_thread_ = std::thread(&TelemetrySubscriber::process_telemetry_loop, this);
}

TelemetrySubscriber::~TelemetrySubscriber() {
    // Stop the processing thread
    running_ = false;
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void TelemetrySubscriber::on_telemetry_ready(std::shared_ptr<TelemetrySnapshot> telemetry) {
    pending_snapshots_.push(std::move(telemetry));
}

void TelemetrySubscriber::process_telemetry_loop() {
    while (running_) {
        auto snapshot = pending_snapshots_.pop();
        if (!snapshot) {
            // No snapshot, sleep a while
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            continue;
        }

        // Merge snapshot into telemetry state
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            telemetry_state_.merge_from(**snapshot);
        }

        // Call the derived class handler
        // Note: We call this with the mutex unlocked to avoid holding it during
        // potentially long operations. Derived classes can lock state_mutex_ if they
        // need to access telemetry_state_ safely.
        on_telemetry_updated(**snapshot);
    }
}
