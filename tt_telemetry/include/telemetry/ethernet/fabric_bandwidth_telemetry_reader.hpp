#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/fabric_bandwidth_telemetry_reader.hpp
 *
 * Helper class to read fabric bandwidth telemetry data from device.
 * Caches the data per update cycle to avoid redundant device reads.
 */

#include <chrono>
#include <cstdint>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <llrt/hal.hpp>
#include <tt_metal/fabric/physical_system_descriptor.hpp>

// Structure to read bandwidth telemetry from device
// Must match LowResolutionBandwidthTelemetryResult on device
struct RiscTimestamp {
    union {
        uint64_t full;
        struct {
            uint32_t lo;
            uint32_t hi;
        };
    };
};

struct LowResolutionBandwidthTelemetryResult {
    RiscTimestamp duration{};
    uint64_t reserved{};
    uint64_t num_words_sent{};
    uint64_t num_packets_sent{};
};

class FabricBandwidthTelemetryReader {
public:
    FabricBandwidthTelemetryReader(
        tt::tt_metal::TrayID tray_id,
        tt::tt_metal::ASICLocation asic_location,
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    // Returns the cached telemetry data, updating from device if needed
    const LowResolutionBandwidthTelemetryResult& get_telemetry(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        std::chrono::steady_clock::time_point start_of_update_cycle);

private:
    void read_from_device(const std::unique_ptr<tt::umd::Cluster>& cluster);

    tt::tt_metal::TrayID tray_id_;
    tt::tt_metal::ASICLocation asic_location_;
    tt::ChipId chip_id_;
    uint32_t channel_;
    tt::umd::CoreCoord ethernet_core_;
    uint32_t bw_telemetry_addr_;
    
    LowResolutionBandwidthTelemetryResult cached_telemetry_;
    std::chrono::steady_clock::time_point last_update_cycle_;
};

