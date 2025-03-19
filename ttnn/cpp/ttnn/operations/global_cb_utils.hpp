// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/global_circular_buffer.hpp"

#include <tt-metalium/global_circular_buffer_impl.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace tt::tt_metal {

// TODO: remove once the multidevice/single device objects are unified
using DeviceGlobalCircularBuffer = std::variant<
    std::monostate,
    tt::tt_metal::experimental::GlobalCircularBuffer,
    ttnn::global_circular_buffer::MultiDeviceGlobalCircularBuffer>;

inline tt::tt_metal::experimental::GlobalCircularBuffer get_global_circular_buffer(
    DeviceGlobalCircularBuffer device_global_cb, chip_id_t device_id) {
    if (std::holds_alternative<tt::tt_metal::experimental::GlobalCircularBuffer>(device_global_cb)) {
        return std::get<tt::tt_metal::experimental::GlobalCircularBuffer>(device_global_cb);
    } else {
        auto& multi_device_global_cb =
            std::get<ttnn::global_circular_buffer::MultiDeviceGlobalCircularBuffer>(device_global_cb);

        for (auto& global_cb_ : multi_device_global_cb.global_circular_buffers) {
            tt::tt_metal::IDevice* global_cb_device = global_cb_.get_device();
            auto global_device_id = global_cb_device->id();
            if (device_id == global_device_id) {
                return global_cb_;
            }
        }

        TT_THROW("Error finding a device for a GlobalCB in MultiDeviceGlobalCircularBuffer");
    }
}

}  // namespace tt::tt_metal
