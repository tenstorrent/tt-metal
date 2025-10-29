// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <bit>
#include <array>

#include <telemetry/ethernet/fabric_bandwidth_telemetry_reader.hpp>

FabricBandwidthTelemetryReader::FabricBandwidthTelemetryReader(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    channel_(channel),
    cached_telemetry_{},
    last_update_cycle_(std::chrono::steady_clock::time_point::min()) {
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    bw_telemetry_addr_ = hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
}

const LowResolutionBandwidthTelemetryResult& FabricBandwidthTelemetryReader::get_telemetry(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    
    // Only read from device if this is a new update cycle
    if (start_of_update_cycle != last_update_cycle_) {
        read_from_device(cluster);
        last_update_cycle_ = start_of_update_cycle;
    }
    
    return cached_telemetry_;
}

void FabricBandwidthTelemetryReader::read_from_device(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    // Read telemetry structure from device
    constexpr size_t telemetry_size = sizeof(LowResolutionBandwidthTelemetryResult);
    std::array<std::byte, telemetry_size> buffer{};
    
    cluster->read_from_device(buffer.data(), chip_id_, ethernet_core_, bw_telemetry_addr_, telemetry_size);
    
    // Convert buffer to telemetry structure
    LowResolutionBandwidthTelemetryResult tel{};
    if (reinterpret_cast<uintptr_t>(buffer.data()) % alignof(LowResolutionBandwidthTelemetryResult) == 0) {
        // Buffer is properly aligned, can use bit_cast
        constexpr size_t NUM_ELEMENTS = 
            ((sizeof(LowResolutionBandwidthTelemetryResult) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        const std::array<uint32_t, NUM_ELEMENTS>& data_array = 
            *reinterpret_cast<const std::array<uint32_t, NUM_ELEMENTS>*>(buffer.data());
        tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(data_array);
    } else {
        // Use memcpy for unaligned access
        std::array<std::byte, sizeof(LowResolutionBandwidthTelemetryResult)> staging_buf{};
        memcpy(staging_buf.data(), buffer.data(), sizeof(LowResolutionBandwidthTelemetryResult));
        tel = std::bit_cast<LowResolutionBandwidthTelemetryResult>(staging_buf);
    }
    
    cached_telemetry_ = tel;
}

