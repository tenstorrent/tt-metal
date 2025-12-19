// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/fabric_telemetry_reader.hpp>

#include <mutex>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <llrt/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <tt_stl/assert.hpp>

FabricTelemetryReader::FabricTelemetryReader(
    tt::ChipId chip_id,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    chip_id_(chip_id),
    cluster_(cluster.get()),
    hal_(hal.get()),
    last_update_cycle_(std::chrono::steady_clock::time_point::min()) {
    TT_FATAL(cluster_ != nullptr, "FabricTelemetryReader: cluster cannot be null");
    TT_FATAL(hal_ != nullptr, "FabricTelemetryReader: hal cannot be null");

    const auto& soc_desc = cluster_->get_soc_descriptor(chip_id_);
    uint32_t num_eth_channels = soc_desc.get_num_eth_channels();
    for (uint32_t channel = 0; channel < num_eth_channels; channel++) {
        cached_telemetry_[channel] = tt::tt_fabric::FabricTelemetrySnapshot{};
    }

    log_debug(
        tt::LogAlways,
        "FabricTelemetryReader initialized for chip {} with {} channels",
        chip_id_,
        cached_telemetry_.size());
}

tt::tt_fabric::FabricTelemetrySnapshot FabricTelemetryReader::read_channel_telemetry(uint32_t channel) {
    try {
        auto snapshot =
            tt::tt_fabric::read_fabric_telemetry(const_cast<tt::umd::Cluster&>(*cluster_), *hal_, chip_id_, channel);

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

const tt::tt_fabric::FabricTelemetrySnapshot* FabricTelemetryReader::get_fabric_telemetry_for_channel(
    uint32_t channel, std::chrono::steady_clock::time_point start_of_update_cycle) {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);

    if (start_of_update_cycle != last_update_cycle_) {
        for (auto& [ch, snapshot] : cached_telemetry_) {
            snapshot = read_channel_telemetry(ch);
        }
        last_update_cycle_ = start_of_update_cycle;
    }

    auto it = cached_telemetry_.find(channel);
    if (it != cached_telemetry_.end()) {
        return &it->second;
    }

    return nullptr;
}
