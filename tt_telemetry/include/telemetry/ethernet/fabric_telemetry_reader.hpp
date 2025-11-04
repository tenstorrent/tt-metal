#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/fabric_telemetry_reader.hpp
 *
 * Helper class to read fabric telemetry data from device.
 * Caches the data per update cycle to avoid redundant device reads.
 */

#include <chrono>
#include <cstdint>
#include <variant>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <llrt/hal.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>
#include <tt_metal/api/tt-metalium/fabric_telemetry.hpp>

using FabricTelemetryContainer =  std::variant<FabricTelemetry<1>, FabricTelemetry<2>>;

class FabricTelemetryReader {
public:
    FabricTelemetryReader(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    // Returns the cached telemetry data, updating from device if needed
    const FabricTelemetryContainer& get_fabric_telemetry(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle);

private:
    void update_telemetry(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle);

    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t fabric_telemetry_addr_;

    FabricTelemetryContainer cached_fabric_telemetry_;
    std::chrono::steady_clock::time_point last_update_cycle_;
};

