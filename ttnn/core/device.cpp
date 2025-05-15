// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include <tt-metalium/device_pool.hpp>

namespace ttnn {

namespace device {

std::shared_ptr<MeshDevice> open_mesh_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    size_t worker_l1_size) {
    return MeshDevice::create_unit_mesh(
        device_id, l1_small_size, trace_region_size, 1, dispatch_core_config, {}, worker_l1_size);
}

void enable_program_cache(IDevice& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(IDevice& device) { device.disable_and_clear_program_cache(); }

void close_device(IDevice& device) {
    // TODO #20966: Remove single device support and branches + dynamic_cast
    if (auto mesh_device = dynamic_cast<MeshDevice*>(&device)) {
        mesh_device->close();
    } else {
        tt::DevicePool::instance().close_device(device.id());
    }
}

bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(IDevice* device) { device->allocator()->deallocate_buffers(); }

}  // namespace device

using namespace device;

}  // namespace ttnn
