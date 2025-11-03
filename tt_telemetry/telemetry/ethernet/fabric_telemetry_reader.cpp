// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstring>
#include <bit>
#include <array>

#include <telemetry/ethernet/fabric_telemetry_reader.hpp>

FabricTelemetryReader::FabricTelemetryReader(
    tt::tt_metal::TrayID tray_id,
    tt::tt_metal::ASICLocation asic_location,
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    tray_id_(tray_id),
    asic_location_(asic_location),
    chip_id_(chip_id),
    cached_bw_telemetry_{},
    cached_wormhole_fabric_telemetry_{},
    cached_blackhole_fabric_telemetry_{},
    last_update_cycle_(std::chrono::steady_clock::time_point::min()) {
    ethernet_core_ = cluster->get_soc_descriptor(chip_id).get_eth_core_for_channel(channel, tt::CoordSystem::LOGICAL);
    fabric_telemetry_addr_ = hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::FABRIC_TELEMETRY);
    bw_telemetry_addr_ = hal->get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::UNRESERVED);
}

// Reads an entire contiguous struct from L1 memory
template <typename TelemetryStruct>
static TelemetryStruct read_from_device(const std::unique_ptr<tt::umd::Cluster>& cluster, tt::ChipId chip_id, tt::umd::CoreCoord ethernet_core, uint32_t l1_addr) {
    // Read telemetry structure from device
    constexpr size_t telemetry_size = sizeof(TelemetryStruct);
    std::array<std::byte, telemetry_size> buffer{};

    cluster->read_from_device(buffer.data(), chip_id, ethernet_core, l1_addr, telemetry_size);

    // Convert buffer to telemetry structure
    TelemetryStruct tel{};
    if (reinterpret_cast<uintptr_t>(buffer.data()) % alignof(TelemetryStruct) == 0) {
        // Buffer is properly aligned, can use bit_cast
        constexpr size_t num_elements = 
            ((sizeof(TelemetryStruct) + sizeof(uint32_t) - 1) / sizeof(uint32_t));
        const std::array<uint32_t, num_elements>& data_array = 
            *reinterpret_cast<const std::array<uint32_t, num_elements>*>(buffer.data());
        tel = std::bit_cast<TelemetryStruct>(data_array);
    } else {
        // Use memcpy for unaligned access
        std::array<std::byte, sizeof(TelemetryStruct)> staging_buf{};
        memcpy(staging_buf.data(), buffer.data(), sizeof(TelemetryStruct));
        tel = std::bit_cast<TelemetryStruct>(staging_buf);
    }

    return tel;
}

void FabricTelemetryReader::update_telemetry(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Only read from device if this is a new update cycle
    if (start_of_update_cycle != last_update_cycle_) {
        cached_bw_telemetry_ = read_from_device<BandwidthTelemetry>(cluster, chip_id_, ethernet_core_, bw_telemetry_addr_);
        switch (cluster->get_cluster_description()->get_arch()) {
        case tt::ARCH::WORMHOLE_B0:
            cached_wormhole_fabric_telemetry_ = read_from_device<FabricTelemetry<1>>(cluster, chip_id_, ethernet_core_, fabric_telemetry_addr_);
            break;
        case tt::ARCH::BLACKHOLE:
            cached_blackhole_fabric_telemetry_ = read_from_device<FabricTelemetry<2>>(cluster, chip_id_, ethernet_core_, fabric_telemetry_addr_);
            break;
        default:
            TT_FATAL(false, "Unknown architecture: {}", cluster->get_cluster_description()->get_arch());
            break;
        }
        last_update_cycle_ = start_of_update_cycle;
    }
}

const BandwidthTelemetry& FabricTelemetryReader::get_bandwidth_telemetry(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    update_telemetry(cluster, start_of_update_cycle);
    return cached_bw_telemetry_;
}

const FabricTelemetry<1>& FabricTelemetryReader::get_wormhole_fabric_telemetry(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    update_telemetry(cluster, start_of_update_cycle);
    return cached_wormhole_fabric_telemetry_;
}

const FabricTelemetry<2>& FabricTelemetryReader::get_blackhole_fabric_telemetry(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    update_telemetry(cluster, start_of_update_cycle);
    return cached_blackhole_fabric_telemetry_;
}