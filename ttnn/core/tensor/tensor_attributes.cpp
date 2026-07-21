// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"

#include <tt-metalium/experimental/distributed_tensor/distributed_tensor_apis.hpp>

namespace ttnn {

using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorTopology;

TensorAttributes::TensorAttributes(HostStorage storage) : storage_(std::move(storage)) {}

TensorAttributes::TensorAttributes(DeviceStorage storage) : storage_(std::move(storage)) {}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }

const TensorSpec& TensorAttributes::get_tensor_spec() const {
    return std::visit(
        ttsl::overloaded{
            [](const HostStorage& host_storage) -> const TensorSpec& {
                return host_storage.host_tensor().tensor_spec();
            },
            [](const DeviceStorage& device_storage) -> const TensorSpec& { return device_storage.get_tensor_spec(); },
        },
        storage_);
}

const TensorTopology& TensorAttributes::get_tensor_topology() const {
    return std::visit(
        ttsl::overloaded{
            [](const HostStorage& host_storage) -> const TensorTopology& {
                return tensor_topology(host_storage.host_tensor());
            },
            [](const DeviceStorage& device_storage) -> const TensorTopology& {
                return device_storage.get_tensor_topology();
            },
        },
        storage_);
}

void TensorAttributes::update_tensor_topology(const TensorTopology& tensor_topology) {
    std::visit(
        ttsl::overloaded{
            [&](HostStorage& host_storage) {
                tt::tt_metal::update_tensor_topology(host_storage.host_tensor(), tensor_topology);
            },
            [&](DeviceStorage& device_storage) {
                tt::tt_metal::update_tensor_topology(device_storage.get_mesh_tensor(), tensor_topology);
            },
        },
        storage_);
}

}  // namespace ttnn
