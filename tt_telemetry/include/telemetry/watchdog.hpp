#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/watchdog.hpp
 *
 * Watchdog thread that monitors the telemetry thread and terminates the process
 * if the telemetry thread becomes unresponsive.
 */

#include <atomic>
#include <memory>
#include <thread>

/**
 * Watchdog class that monitors telemetry thread activity.
 *
 * The watchdog thread periodically checks if a heartbeat counter has been updated.
 * If the counter hasn't changed within the timeout period, it terminates the process.
 *
 * The watchdog thread is automatically started in the constructor if enabled (timeout > 0)
 * and automatically stopped in the destructor.
 */
class Watchdog {
public:
    /**
     * Constructor - automatically starts watchdog thread if enabled
     * @param timeout_seconds Watchdog timeout in seconds. If <= 0, watchdog is disabled.
     */
    explicit Watchdog(int timeout_seconds);

    /**
     * Destructor - stops the watchdog thread if running
     */
    ~Watchdog();

    /**
     * Increment the heartbeat counter - call this from the monitored thread
     */
    void heartbeat();

    /**
     * Check if watchdog is enabled
     */
    bool is_enabled() const { return timeout_seconds_ > 0; }

    /**
     * Get the timeout in seconds
     */
    int timeout_seconds() const { return timeout_seconds_; }

private:
    void stop();
    void watchdog_thread_function();

    int timeout_seconds_;
    std::atomic<uint64_t> heartbeat_counter_;
    std::atomic<bool> stop_requested_;
    std::unique_ptr<std::thread> thread_;
};
