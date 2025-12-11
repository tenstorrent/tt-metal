// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/caching_fabric_telemetry_reader.hpp>

#include <mutex>
#include <unordered_map>

#include <tt-logger/tt-logger.hpp>
#include <llrt/hal.hpp>
#include <tt-metalium/experimental/fabric/fabric_telemetry_reader.hpp>
#include <tt_stl/assert.hpp>

CachingFabricTelemetryReader::CachingFabricTelemetryReader(
    tt::ChipId chip_id,
    uint32_t channel,
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::Hal>& hal) :
    chip_id_(chip_id),
    channel_(channel),
    cluster_(cluster.get()),
    hal_(hal.get()),
    cached_telemetry_(tt::tt_fabric::FabricTelemetrySnapshot{}),
    last_update_cycle_(std::chrono::steady_clock::time_point::min()) {
    TT_FATAL(cluster_ != nullptr, "CachingFabricTelemetryReader: cluster cannot be null");
    TT_FATAL(hal_ != nullptr, "CachingFabricTelemetryReader: hal cannot be null");

    log_debug(tt::LogAlways, "CachingFabricTelemetryReader initialized for chip {}, channel {}", chip_id, channel);
}

tt::tt_fabric::FabricTelemetrySnapshot CachingFabricTelemetryReader::read_telemetry() {
    try {
        auto snapshot =
            tt::tt_fabric::read_fabric_telemetry(const_cast<tt::umd::Cluster&>(*cluster_), *hal_, chip_id_, channel_);

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
            channel_,
            e.what());

        tt::tt_fabric::FabricTelemetrySnapshot empty_snapshot;
        empty_snapshot.dynamic_info.reset();
        return empty_snapshot;
    }
}

const tt::tt_fabric::FabricTelemetrySnapshot* CachingFabricTelemetryReader::get_telemetry(
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);

    if (start_of_update_cycle != last_update_cycle_) {
        cached_telemetry_ = read_telemetry();
        last_update_cycle_ = start_of_update_cycle;
    }

    return &cached_telemetry_;
}
