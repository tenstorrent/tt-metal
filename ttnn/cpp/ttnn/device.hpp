// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// #include "tt_dnn/op_library/eltwise_binary/eltwise_binary_op.hpp"
// #include "tt_eager/tensor/tensor_utils.hpp"
// #include "tt_eager/tt_dnn/op_library/bcast/bcast_op.hpp"
// #include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace device {
using Device = ttnn::Device;

// TODO : Circle back and update for Multi-Device
inline std::vector<Device *> _devices;

inline Device &open_device(int device_id) {
    auto num_devices = tt::tt_metal::GetNumAvailableDevices();
    _devices.resize(num_devices, nullptr);
    TT_ASSERT(device_id < num_devices);
    if (_devices[device_id] == nullptr) {
        _devices[device_id] = CreateDevice(device_id, 1);
    }
    return *_devices[device_id];
}

inline void close_device(Device &device) {
    TT_ASSERT(device.id() < _devices.size());
    size_t offset = device.id();
    if (_devices[offset] != nullptr) {
        _devices[offset]->close();
        _devices[offset] = nullptr;
    }
}

}  // namespace device

using namespace device;

}  // namespace ttnn
