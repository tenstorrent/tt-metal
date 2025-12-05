#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/fabric_telemetry_reader.hpp
 *
 * Caching wrapper for fabric telemetry data.
 * Reads from device once per update cycle and caches results to avoid
 * redundant reads when multiple metrics need the same data.
 */

#include <chrono>
#include <vector>

#include <tt-metalium/types.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

class FabricTelemetryReader {
public:
    FabricTelemetryReader(tt::ChipId chip_id);

    // Returns cached telemetry data for all channels, updating from device if needed
    const std::vector<tt::tt_fabric::FabricTelemetrySample>& get_fabric_telemetry(
        std::chrono::steady_clock::time_point start_of_update_cycle);

private:
    void update_telemetry(std::chrono::steady_clock::time_point start_of_update_cycle);

    tt::ChipId chip_id_;
    std::vector<tt::tt_fabric::FabricTelemetrySample> cached_telemetry_;
    std::chrono::steady_clock::time_point last_update_cycle_;
};
