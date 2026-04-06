// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(HostStorage storage) : storage_(std::move(storage)) {}

TensorAttributes::TensorAttributes(DeviceStorage storage) : storage_(std::move(storage)) {}

// Transitional: assumes a HostStorage constructed without proper TensorSpec and TensorTopology.
// Overrides the existing spec and topology in the HostStorage.
TensorAttributes::TensorAttributes(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
    storage_(HostStorage(std::move(storage), std::move(tensor_spec), std::move(tensor_topology))) {}

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
                return host_storage.host_tensor().tensor_topology();
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
            [&](HostStorage& host_storage) { host_storage.host_tensor().update_tensor_topology(tensor_topology); },
            [&](DeviceStorage& device_storage) { device_storage.update_tensor_topology(tensor_topology); },
        },
        storage_);
}

}  // namespace tt::tt_metal
