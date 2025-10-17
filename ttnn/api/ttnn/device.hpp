// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"

namespace ttnn {

namespace device {

using IDevice = ttnn::IDevice;
using MeshDevice = tt::tt_metal::distributed::MeshDevice;

std::shared_ptr<MeshDevice> open_mesh_device(
    int device_id,
    size_t l1_small_size = DEFAULT_L1_SMALL_SIZE,
    size_t trace_region_size = DEFAULT_TRACE_REGION_SIZE,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config = tt::tt_metal::DispatchCoreConfig{},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);
void close_device(IDevice& device);
void enable_program_cache(IDevice& device);
void disable_and_clear_program_cache(IDevice& device);
bool is_wormhole_or_blackhole(tt::ARCH arch);
void deallocate_buffers(IDevice* device);

}  // namespace device

using namespace device;

}  // namespace ttnn
