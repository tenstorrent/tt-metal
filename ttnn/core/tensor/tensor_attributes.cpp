// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"

namespace ttnn {

TensorAttributes::TensorAttributes(ttnn::HostStorage storage) : storage_(std::move(storage)) {}

TensorAttributes::TensorAttributes(ttnn::DeviceStorage storage) : storage_(std::move(storage)) {}

const ttnn::Storage& TensorAttributes::get_storage() const { return storage_; }
ttnn::Storage& TensorAttributes::get_storage() { return storage_; }

const tt::tt_metal::TensorSpec& TensorAttributes::get_tensor_spec() const {
    return std::visit(
        ttsl::overloaded{
            [](const HostStorage& host_storage) -> const tt::tt_metal::TensorSpec& {
                return host_storage.host_tensor().tensor_spec();
            },
            [](const DeviceStorage& device_storage) -> const tt::tt_metal::TensorSpec& {
                return device_storage.get_tensor_spec();
            },
        },
        storage_);
}

const tt::tt_metal::TensorTopology& TensorAttributes::get_tensor_topology() const {
    return std::visit(
        ttsl::overloaded{
            [](const HostStorage& host_storage) -> const tt::tt_metal::TensorTopology& {
                return host_storage.host_tensor().tensor_topology();
            },
            [](const DeviceStorage& device_storage) -> const tt::tt_metal::TensorTopology& {
                return device_storage.get_tensor_topology();
            },
        },
        storage_);
}

void TensorAttributes::update_tensor_topology(const tt::tt_metal::TensorTopology& tensor_topology) {
    std::visit(
        ttsl::overloaded{
            [&](ttnn::HostStorage& host_storage) {
                host_storage.host_tensor().update_tensor_topology(tensor_topology);
            },
            [&](ttnn::DeviceStorage& device_storage) {
                device_storage.get_mesh_tensor().update_tensor_topology(tensor_topology);
            },
        },
        storage_);
}

}  // namespace ttnn
