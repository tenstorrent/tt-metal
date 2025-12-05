// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/fabric_telemetry_reader.hpp>

#include <tt-metalium/experimental/fabric/control_plane.hpp>
#include <tt_metal/impl/context/metal_context.hpp>

FabricTelemetryReader::FabricTelemetryReader(tt::ChipId chip_id) :
    chip_id_(chip_id), cached_telemetry_{}, last_update_cycle_(std::chrono::steady_clock::time_point::min()) {}

void FabricTelemetryReader::update_telemetry(std::chrono::steady_clock::time_point start_of_update_cycle) {
    // Only read from device if this is a new update cycle
    if (start_of_update_cycle != last_update_cycle_) {
        auto& metal_ctx = tt::tt_metal::MetalContext::instance();
        auto& control_plane = metal_ctx.get_control_plane();
        const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(chip_id_);
        cached_telemetry_ = tt::tt_fabric::read_fabric_telemetry(fabric_node_id);
        last_update_cycle_ = start_of_update_cycle;
    }
}

const std::vector<tt::tt_fabric::FabricTelemetrySample>& FabricTelemetryReader::get_fabric_telemetry(
    std::chrono::steady_clock::time_point start_of_update_cycle) {
    update_telemetry(start_of_update_cycle);
    return cached_telemetry_;
}
