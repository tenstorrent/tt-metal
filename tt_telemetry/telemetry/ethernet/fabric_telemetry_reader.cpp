// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/fabric_telemetry_reader.hpp>

#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <llrt/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>

FabricTelemetryReader::FabricTelemetryReader(
    tt::ChipId chip_id,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    chip_id_(chip_id),
    cluster_(cluster.get()),
    hal_(hal.get()),
    last_update_cycle_(std::chrono::steady_clock::time_point::min()) {
    const auto& soc_desc = cluster_->get_soc_descriptor(chip_id_);
    for (const auto& [logical_core, channel] : soc_desc.logical_eth_core_to_chan_map) {
        cached_telemetry_[static_cast<uint32_t>(channel)] = tt::tt_fabric::FabricTelemetrySnapshot{};
    }

    log_debug(
        tt::LogAlways,
        "FabricTelemetryReader initialized for chip {} with {} channels",
        chip_id_,
        cached_telemetry_.size());
}

tt::tt_fabric::FabricTelemetrySnapshot FabricTelemetryReader::read_channel_telemetry(uint32_t channel) {
    try {
        auto snapshot = tt::tt_fabric::read_fabric_telemetry(*cluster_, *hal_, chip_id_, channel);

        if (snapshot.static_info.supported_stats == 0) {
            log_debug(
                tt::LogAlways,
                "Fabric telemetry disabled for chip {} channel {} (supported_stats=0). "
                "Set TT_METAL_FABRIC_TELEMETRY=1 to enable.",
                chip_id_,
                channel);
            snapshot.dynamic_info.reset();
        }

        return snapshot;

    } catch (const std::exception& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read fabric telemetry for chip {} channel {}: {}. "
            "Device may be busy or unavailable.",
            chip_id_,
            channel,
            e.what());

        tt::tt_fabric::FabricTelemetrySnapshot empty_snapshot;
        empty_snapshot.dynamic_info.reset();
        return empty_snapshot;
    }
}

void FabricTelemetryReader::update_telemetry(std::chrono::steady_clock::time_point start_of_update_cycle) {
    if (start_of_update_cycle != last_update_cycle_) {
        for (auto& [channel, snapshot] : cached_telemetry_) {
            snapshot = read_channel_telemetry(channel);
        }
        last_update_cycle_ = start_of_update_cycle;
    }
}

const tt::tt_fabric::FabricTelemetrySnapshot* FabricTelemetryReader::get_fabric_telemetry_for_channel(
    uint32_t channel, std::chrono::steady_clock::time_point start_of_update_cycle) {
    update_telemetry(start_of_update_cycle);

    auto it = cached_telemetry_.find(channel);
    if (it != cached_telemetry_.end()) {
        return &it->second;
    }

    return nullptr;
}
