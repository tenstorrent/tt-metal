// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/creation.hpp"

#include <optional>

#include "ttnn/any_device.hpp"

namespace ttnn::operations::creation::detail {

OptionalAnyDevice::OptionalAnyDevice(std::nullopt_t) {}
OptionalAnyDevice::OptionalAnyDevice(ttnn::AnyDevice device) : device_(std::make_optional<ttnn::AnyDevice>(device)) {}

// TODO: some of these won't be needed, as we unify the APIs.
OptionalAnyDevice::OptionalAnyDevice(const std::optional<std::reference_wrapper<tt::tt_metal::IDevice>>& device) :
    device_(device.has_value() ? std::make_optional<AnyDevice>(&device->get()) : std::nullopt) {}
OptionalAnyDevice::OptionalAnyDevice(
    const std::optional<std::reference_wrapper<tt::tt_metal::distributed::MeshDevice>>& mesh_device) :
    device_(mesh_device.has_value() ? std::make_optional<AnyDevice>(&mesh_device->get()) : std::nullopt) {}
OptionalAnyDevice::OptionalAnyDevice(std::reference_wrapper<tt::tt_metal::IDevice> device) :
    device_(std::make_optional<AnyDevice>(&device.get())) {}
OptionalAnyDevice::OptionalAnyDevice(std::reference_wrapper<tt::tt_metal::distributed::MeshDevice> mesh_device) :
    device_(std::make_optional<AnyDevice>(&mesh_device.get())) {}

OptionalAnyDevice::OptionalAnyDevice(tt::tt_metal::IDevice& device) : device_(std::make_optional<AnyDevice>(&device)) {}
OptionalAnyDevice::OptionalAnyDevice(tt::tt_metal::distributed::MeshDevice& mesh_device) :
    device_(std::make_optional<AnyDevice>(&mesh_device)) {}

}  // namespace ttnn::operations::creation::detail
