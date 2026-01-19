#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/caching_telemetry_reader.hpp
 *
 * Caching reader for telemetry data. Reads from device L1 memory once per update cycle and
 * caches results to avoid redundant reads when multiple metrics need the same data.
 *
 * THREAD SAFETY REQUIREMENTS:
 * - Designed for single-threaded sequential metric updates within one update cycle
 * - All metrics must be updated from the same thread using the same start_of_update_cycle timestamp
 * - The telemetry collector enforces this by calling metric->update() sequentially
 * - DO NOT call get_telemetry() concurrently from multiple threads
 * - DO NOT cache the returned pointer beyond the immediate usage
 *
 * USAGE PATTERN (enforced by telemetry collector):
 * 1. Thread 1: metric_a.update(cluster, timestamp_T) → calls get_telemetry(timestamp_T) → reads device
 * 2. Thread 1: metric_b.update(cluster, timestamp_T) → calls get_telemetry(timestamp_T) → returns cached
 * 3. Thread 1: metric_c.update(cluster, timestamp_T) → calls get_telemetry(timestamp_T) → returns cached
 *
 * The returned pointer is valid only until the next call to get_telemetry with a different timestamp.
 */

#include <chrono>
#include <memory>
#include <mutex>

#include <tt_stl/assert.hpp>

template <typename TelemetrySnapshot>
class CachingTelemetryReader {
public:
    CachingTelemetryReader() :
        cached_telemetry_(TelemetrySnapshot{}), last_update_cycle_(std::chrono::steady_clock::time_point::min()) {}

    virtual ~CachingTelemetryReader() = default;

    CachingTelemetryReader(const CachingTelemetryReader&) = delete;
    CachingTelemetryReader& operator=(const CachingTelemetryReader&) = delete;
    CachingTelemetryReader(CachingTelemetryReader&&) = delete;
    CachingTelemetryReader& operator=(CachingTelemetryReader&&) = delete;

    // Returns cached telemetry snapshot for the given update cycle.
    // Updates from device if this is a new update cycle (cache miss).
    //
    // IMPORTANT: The returned pointer is valid only until the next call to this method.
    // Callers must use the data immediately and not store the pointer.
    //
    // Thread safety: Safe only when called sequentially from a single thread with the same
    // start_of_update_cycle timestamp (as enforced by the telemetry collector).
    const TelemetrySnapshot* get_telemetry(std::chrono::steady_clock::time_point start_of_update_cycle) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (start_of_update_cycle != last_update_cycle_) {
            cached_telemetry_ = read_telemetry();
            last_update_cycle_ = start_of_update_cycle;
        }
        return &cached_telemetry_;
    }

protected:
    // Reads telemetry directly from L1
    virtual TelemetrySnapshot read_telemetry() = 0;

private:
    TelemetrySnapshot cached_telemetry_;
    std::chrono::steady_clock::time_point last_update_cycle_;
    std::mutex mtx_;
};
