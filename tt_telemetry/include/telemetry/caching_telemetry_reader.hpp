#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/caching_telemetry_reader.hpp
 *
 * Caching reader for fabric telemetry data. Reads from device L1 memory once per update cycle and
 * caches results to avoid redundant reads when multiple metrics need the same data.
 */

#include <chrono>
#include <memory>
#include <mutex>

#include <tt_stl/assert.hpp>

template <typename TelemetryContainer>
class CachingTelemetryReader {
public:
    CachingTelemetryReader() :
        cached_telemetry_(TelemetryContainer{}), last_update_cycle_(std::chrono::steady_clock::time_point::min()) {}

    virtual ~CachingTelemetryReader() {}

    CachingTelemetryReader(const CachingTelemetryReader&) = delete;
    CachingTelemetryReader& operator=(const CachingTelemetryReader&) = delete;
    CachingTelemetryReader(CachingTelemetryReader&&) = delete;
    CachingTelemetryReader& operator=(CachingTelemetryReader&&) = delete;

    // Returns cached telemetry snapshot. Updates from device if this is a new update cycle.
    // Returns nullptr if telemetry unavailable. Note: Returned pointer is valid only until next
    // call to this method
    const TelemetryContainer* get_telemetry(std::chrono::steady_clock::time_point start_of_update_cycle) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (start_of_update_cycle != last_update_cycle_) {
            cached_telemetry_ = read_telemetry();
            last_update_cycle_ = start_of_update_cycle;
        }
        return &cached_telemetry_;
    }

protected:
    // Reads telemetry directly from L1
    virtual TelemetryContainer read_telemetry() = 0;

private:
    TelemetryContainer cached_telemetry_;
    std::chrono::steady_clock::time_point last_update_cycle_;
    mutable std::mutex mtx_;
};
