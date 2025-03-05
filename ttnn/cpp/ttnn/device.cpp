// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include <tt-metalium/device_pool.hpp>

namespace ttnn {

namespace device {

std::shared_ptr<MeshDevice> open_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config) {
    return MeshDevice::create_single_device(device_id, l1_small_size, trace_region_size, 1, dispatch_core_config);
}

bool is_device_open(int device_id) { return tt::DevicePool::instance().is_device_active(device_id); }

void enable_program_cache(MeshDevice& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(MeshDevice& device) { device.disable_and_clear_program_cache(); }

void close_device(MeshDevice& device) { device.close(); }

bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(MeshDevice* device) {
    device->push_work([device]() mutable { device->allocator()->deallocate_buffers(); }, false);
}

}  // namespace device

using namespace device;

}  // namespace ttnn
