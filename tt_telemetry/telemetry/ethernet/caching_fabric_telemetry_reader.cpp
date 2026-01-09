// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/caching_fabric_telemetry_reader.hpp>

#include <mutex>
#include <typeinfo>
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
    chip_id_(chip_id), channel_(channel), cluster_(cluster.get()), hal_(hal.get()) {
    TT_FATAL(cluster_ != nullptr, "CachingFabricTelemetryReader: cluster cannot be null");
    TT_FATAL(hal_ != nullptr, "CachingFabricTelemetryReader: hal cannot be null");
    log_debug(tt::LogAlways, "CachingFabricTelemetryReader initialized for chip {}, channel {}", chip_id, channel);
}

tt::tt_fabric::FabricTelemetrySnapshot CachingFabricTelemetryReader::read_telemetry() {
    try {
        auto snapshot = tt::tt_fabric::read_fabric_telemetry(*cluster_, *hal_, chip_id_, channel_);

        if (snapshot.static_info.supported_stats == 0) {
            log_debug(
                tt::LogAlways,
                "Fabric telemetry disabled for chip {} channel {} (supported_stats=0). "
                "Set TT_METAL_FABRIC_TELEMETRY=\"chips=all;eth=all;erisc=all;stats=BANDWIDTH|...\" "
                "and TT_METAL_FABRIC_BW_TELEMETRY=1 to enable.",
                chip_id_,
                channel_);
            snapshot.dynamic_info.reset();
        }

        return snapshot;

    } catch (const std::runtime_error& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read fabric telemetry for chip {} channel {} (runtime_error): {}. "
            "Device may be busy or unavailable.",
            chip_id_,
            channel_,
            e.what());

        tt::tt_fabric::FabricTelemetrySnapshot empty_snapshot;
        empty_snapshot.dynamic_info.reset();
        return empty_snapshot;

    } catch (const std::exception& e) {
        log_warning(
            tt::LogAlways,
            "Failed to read fabric telemetry for chip {} channel {} ({}): {}. "
            "Unexpected error.",
            chip_id_,
            channel_,
            typeid(e).name(),
            e.what());

        tt::tt_fabric::FabricTelemetrySnapshot empty_snapshot;
        empty_snapshot.dynamic_info.reset();
        return empty_snapshot;

    } catch (...) {
        log_warning(
            tt::LogAlways,
            "Failed to read fabric telemetry for chip {} channel {}: Unknown exception type. Unexpected error.",
            chip_id_,
            channel_);

        tt::tt_fabric::FabricTelemetrySnapshot empty_snapshot;
        empty_snapshot.dynamic_info.reset();
        return empty_snapshot;
    }
}
