// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"
#include "tt_metal/impl/device/device_pool.hpp"

namespace ttnn {

namespace device {

Device &open_device(int device_id, size_t l1_small_size, size_t trace_region_size, tt::tt_metal::DispatchCoreType dispatch_core_type) {
    tt::DevicePool::initialize({device_id}, 1, l1_small_size, trace_region_size, dispatch_core_type, {});
    return *(tt::DevicePool::instance().get_active_device(device_id));
}

bool is_device_open(int device_id){
    return tt::DevicePool::instance().is_device_active(device_id);
}

void enable_program_cache(Device &device) {
    DeviceEnableProgramCache(&device);
}

void disable_and_clear_program_cache(Device &device) {
    DeviceDisableAndClearProgramCache(&device);
}

void close_device(Device &device) {
    tt::DevicePool::instance().close_device(DeviceId(&device));

}

bool is_wormhole_or_blackhole(tt::ARCH arch) {
    return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE;
}

void deallocate_buffers(Device* device) {
        DevicePushWork(device, [device] () mutable {
            DeviceDeallocateBuffers(device);
        });
}

}  // namespace device

using namespace device;

}  // namespace ttnn
