// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>

#include "tt_metal/distributed/mesh_device.hpp"
#include "tt_metal/impl/device/device.hpp"

namespace ttnn {

// SimpleDevice is a wrapper around Device / MeshDevice to use in interfaces that can accept either.
// This class is cheaply copyable, use value semantics to pass it around.
//
// TODO: the eventual goal is to lower this primitive into tt_metal. In the long term, we also want to extend the
// functionality with the "distributed device" semantics.
class SimpleDevice {
   public:
    // Allow implicit conversion for transparent migration.
    // Expect the pointers to be non-null, and remain valid for the lifetime of SimpleDevice.
    SimpleDevice(tt::tt_metal::Device* device) : metal_device_{device} {}
    SimpleDevice(tt::tt_metal::distributed::MeshDevice* mesh_device) : metal_device_{mesh_device} {}
    SimpleDevice(const SimpleDevice&) = default;
    SimpleDevice& operator=(const SimpleDevice&) = default;
    SimpleDevice(SimpleDevice&&) = delete;
    SimpleDevice& operator=(SimpleDevice&&) = delete;

    std::vector<tt::tt_metal::Device*> get_devices() {
        if (auto* device = std::get_if<tt::tt_metal::Device*>(&metal_device_); device != nullptr) {
            return {*device};
        } else {
            return std::get<tt::tt_metal::distributed::MeshDevice*>(metal_device_)->get_devices();
        }
    }

   private:
    std::variant<tt::tt_metal::Device*, tt::tt_metal::distributed::MeshDevice*> metal_device_;
};

}  // namespace ttnn
