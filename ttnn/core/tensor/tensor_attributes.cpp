// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec, TensorTopology tensor_topology) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)), tensor_topology_(std::move(tensor_topology)) {
    if (auto* host_storage = std::get_if<HostStorage>(&storage_)) {
        storage_ = HostStorage(std::move(*host_storage), tensor_spec_, tensor_topology_);
    }
}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }

const TensorSpec& TensorAttributes::get_tensor_spec() const {
    if (const auto* host_storage = std::get_if<HostStorage>(&storage_)) {
        return host_storage->host_tensor().tensor_spec();
    }
    return tensor_spec_;
}

const TensorTopology& TensorAttributes::get_tensor_topology() const {
    if (const auto* host_storage = std::get_if<HostStorage>(&storage_)) {
        return host_storage->host_tensor().tensor_topology();
    }
    return tensor_topology_;
}

TensorAttributes TensorAttributes::with_tensor_topology(TensorTopology tensor_topology) const {
    // This need to moved to HostTensor after TODO(#39485)
    if (const auto* host_storage = std::get_if<HostStorage>(&storage_)) {
        HostStorage new_storage(*host_storage, tensor_spec_, tensor_topology);
        return TensorAttributes(std::move(new_storage), tensor_spec_, tensor_topology);
    }
    TensorAttributes result = *this;
    result.tensor_topology_ = std::move(tensor_topology);
    return result;
}

}  // namespace tt::tt_metal
