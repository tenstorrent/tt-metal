#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/caching_fabric_telemetry_reader.hpp
 *
 * Caching wrapper for fabric telemetry data.
 * Reads from device L1 memory once per update cycle and caches results to avoid
 * redundant reads when multiple metrics need the same data.
 *
 * ARCHITECTURE:
 * This class minimizes Metal code dependencies by reading L1 directly via UMD cluster.
 * It does NOT use MetalContext or call Metal functions like read_fabric_telemetry().
 * Instead, it:
 *   1. Gets L1 telemetry address from HAL
 *   2. Reads raw bytes from device via cluster->read_from_device()
 *   3. Parses using HAL factory views and converter templates
 *
 * Fabric node ID (mesh_id, device_id) is read from the telemetry data itself,
 * not looked up via control plane.
 */

#include <chrono>
#include <memory>
#include <mutex>
#include <unordered_map>

#include <tt-metalium/experimental/fabric/fabric_telemetry.hpp>
#include <umd/device/cluster.hpp>

#include <telemetry/caching_telemetry_reader.hpp>

// Forward declarations
namespace tt::tt_metal {
class Hal;
}

class CachingFabricTelemetryReader final : public CachingTelemetryReader<tt::tt_fabric::FabricTelemetrySnapshot> {
public:
    // Constructor takes references available from create_ethernet_metrics()
    // Note: cluster and hal must outlive this object (guaranteed by telemetry collector lifecycle)
    CachingFabricTelemetryReader(
        tt::ChipId chip_id,
        uint32_t channel,
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::Hal>& hal);

    CachingFabricTelemetryReader(const CachingFabricTelemetryReader&) = delete;
    CachingFabricTelemetryReader& operator=(const CachingFabricTelemetryReader&) = delete;
    CachingFabricTelemetryReader(CachingFabricTelemetryReader&&) = delete;
    CachingFabricTelemetryReader& operator=(CachingFabricTelemetryReader&&) = delete;

private:
    tt::tt_fabric::FabricTelemetrySnapshot read_telemetry() override;

    const tt::ChipId chip_id_;
    const uint32_t channel_;
    tt::umd::Cluster* cluster_;
    tt::tt_metal::Hal* hal_;
};
