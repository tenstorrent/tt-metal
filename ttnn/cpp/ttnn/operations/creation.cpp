// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation.hpp"

#include <optional>

#include "ttnn/simple_device.hpp"

namespace ttnn::operations::creation::detail {

OptionalSimpleDevice::OptionalSimpleDevice(std::nullopt_t) {}
OptionalSimpleDevice::OptionalSimpleDevice(ttnn::SimpleDevice device) :
    device_(std::make_optional<ttnn::SimpleDevice>(device)) {}

// TODO: some of these won't be needed, as we unify the APIs.
OptionalSimpleDevice::OptionalSimpleDevice(const std::optional<std::reference_wrapper<tt::tt_metal::Device>>& device) :
    device_(device.has_value() ? std::make_optional<SimpleDevice>(&device->get()) : std::nullopt) {}
OptionalSimpleDevice::OptionalSimpleDevice(
    const std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice>>& mesh_device) :
    device_(mesh_device.has_value() ? std::make_optional<SimpleDevice>(&mesh_device->get()) : std::nullopt) {}
OptionalSimpleDevice::OptionalSimpleDevice(std::reference_wrapper<tt::tt_metal::Device> device) :
    device_(std::make_optional<SimpleDevice>(&device.get())) {}
OptionalSimpleDevice::OptionalSimpleDevice(std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> mesh_device) :
    device_(std::make_optional<SimpleDevice>(&mesh_device.get())) {}

OptionalSimpleDevice::OptionalSimpleDevice(tt::tt_metal::Device& device) :
    device_(std::make_optional<SimpleDevice>(&device)) {}
OptionalSimpleDevice::OptionalSimpleDevice(tt::tt_metal::distributed::MeshDevice& mesh_device) :
    device_(std::make_optional<SimpleDevice>(&mesh_device)) {}

}  // namespace ttnn::operations::creation::detail
