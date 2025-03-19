// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/device.hpp>

namespace ttnn {

// AnyDevice is a wrapper around Device / MeshDevice to use in interfaces that can accept either.
// This class is cheaply copyable, use value semantics to pass it around.
//
// TODO: the eventual goal is to lower this primitive into tt_metal. In the long term, we also want to extend the
// functionality with the "distributed device" semantics.
class AnyDevice {
public:
    // Allow implicit conversion for transparent migration.
    // Expect the pointers to be non-null, and remain valid for the lifetime of AnyDevice.
    AnyDevice(tt::tt_metal::IDevice* device) : metal_device_{device} {}
    AnyDevice(tt::tt_metal::distributed::MeshDevice* mesh_device) : metal_device_{mesh_device} {}
    AnyDevice(const AnyDevice&) = default;
    AnyDevice& operator=(const AnyDevice&) = default;
    AnyDevice(AnyDevice&&) = delete;
    AnyDevice& operator=(AnyDevice&&) = delete;

    std::vector<tt::tt_metal::IDevice*> get_devices() {
        if (auto* device = std::get_if<tt::tt_metal::IDevice*>(&metal_device_); device != nullptr) {
            return {*device};
        } else {
            return std::get<tt::tt_metal::distributed::MeshDevice*>(metal_device_)->get_devices();
        }
    }

private:
    std::variant<tt::tt_metal::IDevice*, tt::tt_metal::distributed::MeshDevice*> metal_device_;
};

}  // namespace ttnn
