// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"

namespace ttnn {

namespace device {

namespace device_pool {

// Definition of the global device vector
std::vector<Device*> devices;

} // device_pool

Device &open_device(int device_id, size_t l1_small_size) {
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    device_pool::devices.resize(num_devices, nullptr);
    TT_ASSERT(device_id < num_devices);
    if (device_pool::devices[device_id] == nullptr) {
        device_pool::devices[device_id] = CreateDevice(device_id, 1, l1_small_size);
    }
    TT_FATAL(
        device_pool::devices[device_id]->get_l1_small_size() == l1_small_size,
        "Device L1 small size mismatch, device already opened with different L1 small size.");
    return *device_pool::devices[device_id];
}

bool is_device_open(int device_id){
    return device_id < device_pool::devices.size() && device_pool::devices[device_id] != nullptr;
}

void enable_program_cache(Device &device) {
    TT_ASSERT(device.id() < device_pool::devices.size());
    TT_ASSERT(device_pool::devices[device.id()] != nullptr);

    device_pool::devices[device.id()]->enable_program_cache();
}

void disable_and_clear_program_cache(Device &device) {
    TT_ASSERT(device.id() < device_pool::devices.size());
    TT_ASSERT(device_pool::devices[device.id()] != nullptr);

    device_pool::devices[device.id()]->disable_and_clear_program_cache();
}

void close_device(Device &device) {
    TT_ASSERT(device.id() < device_pool::devices.size());

    size_t offset = device.id();
    if (device_pool::devices[offset] != nullptr) {
        tt::tt_metal::detail::DeallocateBuffers(device_pool::devices[offset]);
        device_pool::devices[offset]->close();
        delete device_pool::devices[offset];
        device_pool::devices[offset] = nullptr;
    }
}

}  // namespace device

using namespace device;

}  // namespace ttnn
