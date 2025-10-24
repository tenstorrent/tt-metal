// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <thread>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <telemetry/watchdog.hpp>

Watchdog::Watchdog(int timeout_seconds) :
    timeout_seconds_(timeout_seconds), heartbeat_counter_(0), stop_requested_(false), thread_(nullptr) {
    // Automatically start watchdog thread if enabled
    if (timeout_seconds_ > 0) {
        thread_ = std::make_unique<std::thread>(&Watchdog::watchdog_thread_function, this);
    }
}

Watchdog::~Watchdog() { stop(); }

void Watchdog::heartbeat() { heartbeat_counter_.fetch_add(1); }

void Watchdog::stop() {
    if (thread_ && thread_->joinable()) {
        log_info(tt::LogAlways, "[Watchdog] Stopping thread");
        stop_requested_.store(true);
        thread_->join();
        thread_.reset();
    }
}

void Watchdog::watchdog_thread_function() {
    uint64_t last_heartbeat = heartbeat_counter_.load();

    while (!stop_requested_.load()) {
        // Sleep for the timeout duration
        std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds_));

        if (stop_requested_.load()) {
            break;
        }

        // Check if heartbeat has advanced
        uint64_t current_heartbeat = heartbeat_counter_.load();

        if (current_heartbeat == last_heartbeat) {
            // Bring down the entire process
            log_fatal(
                tt::LogAlways,
                "[Watchdog] Timeout: Telemetry thread has not advanced in {} seconds. Last heartbeat: {}. "
                "The telemetry collection loop appears to be stuck or deadlocked. This is often due to a device being "
                "reset.",
                timeout_seconds_,
                last_heartbeat);
            exit(1);
        }

        // Update last known heartbeat
        last_heartbeat = current_heartbeat;
    }

    log_info(tt::LogAlways, "[Watchdog] Monitoring thread stopped");
}
