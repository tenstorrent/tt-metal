// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/dispatch_core_common.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>

#include "ttnn/types.hpp"

namespace ttnn {

namespace device {

using IDevice = ttnn::IDevice;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

// Build a cluster/arch-aware DispatchCoreConfig. Any field left unspecified is resolved from
// the active cluster/arch:
//   type:  ETH for N300/T3K/N300_2x2 clusters; otherwise WORKER.
//   axis:  Blackhole without fabric tensix MUX -> COL; otherwise ROW.
// Throws if type == ETH and axis == COL, or if axis == ROW on Blackhole without fabric tensix MUX.
tt::tt_metal::DispatchCoreConfig create_cluster_aware_dispatch_config(
    std::optional<tt::tt_metal::DispatchCoreType> type = std::nullopt,
    std::optional<tt::tt_metal::DispatchCoreAxis> axis = std::nullopt,
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config = tt::tt_fabric::FabricTensixConfig::DISABLED);

std::shared_ptr<MeshDevice> open_mesh_device(
    int device_id,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    size_t num_command_queues = 1,
    const std::optional<tt::tt_metal::DispatchCoreConfig>& dispatch_core_config = std::nullopt,
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
void close_device(MeshDevice& device);
void enable_program_cache(IDevice& device);
void disable_and_clear_program_cache(IDevice& device);
bool is_wormhole_or_blackhole(tt::ARCH arch);
void deallocate_buffers(IDevice* device);

/**
 * Sets the default device to be used for auto-formatting operations
 * @param dev Pointer to the device to be used
 * @note This functionality is planned for deprecation in the future.
 */
void SetDefaultDevice(MeshDevice* dev);

/**
 * Gets the default device used for auto-formatting operations
 * @return Pointer to the default device
 * @note This functionality is planned for deprecation in the future.
 */
MeshDevice* GetDefaultDevice();

}  // namespace device

using namespace device;

}  // namespace ttnn
