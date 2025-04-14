// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include <tt-metalium/device_pool.hpp>

namespace ttnn {

namespace device {

IDevice& open_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    size_t worker_l1_size) {
    tt::DevicePool::initialize(
        {device_id}, 1, l1_small_size, trace_region_size, dispatch_core_config, {}, worker_l1_size);
    return *(tt::DevicePool::instance().get_active_device(device_id));
}

bool is_device_open(int device_id) { return tt::DevicePool::instance().is_device_active(device_id); }

void enable_program_cache(IDevice& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(IDevice& device) { device.disable_and_clear_program_cache(); }

void close_device(IDevice& device) { tt::DevicePool::instance().close_device(device.id()); }

bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(IDevice* device) {
    device->push_work([device]() mutable { device->allocator()->deallocate_buffers(); });
}

}  // namespace device

using namespace device;

}  // namespace ttnn
