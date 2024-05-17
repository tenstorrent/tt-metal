// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"

namespace ttnn {

namespace device {

Device &open_device(int device_id, size_t l1_small_size) {
    tt::DevicePool::initialize({device_id}, 1, l1_small_size);
    return *(tt::DevicePool::instance().get_active_device(device_id));
}

bool is_device_open(int device_id){
    return tt::DevicePool::instance().is_device_active(device_id);
}

void enable_program_cache(Device &device) {
    device.enable_program_cache();
}

void disable_and_clear_program_cache(Device &device) {
    device.disable_and_clear_program_cache();
}

void close_device(Device &device) {
    tt::DevicePool::instance().close_device(device.id());
}

}  // namespace device

using namespace device;

}  // namespace ttnn
