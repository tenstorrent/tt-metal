// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/distributed/tensor_topology.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(HostStorage storage);

    // Transitional constructor: use TensorAttributes(HostStorage) instead.
    //
    // Accepts a pre-transition HostStorage (constructed without TensorSpec and
    // TensorTopology) and assigns them during TensorAttributes construction.
    // Overrides any existing spec/topology in the HostStorage.
    //
    // e.g. This protects this usage:
    // HostStorage storage(buffer);
    // Tensor(storage, tensor_spec, tensor_topology);
    //
    // (after transition, should be):
    // HostStorage storage(HostTensor(buffer, tensor_spec, tensor_topology));
    // Tensor(storage);
    //
    // TODO(#40348): Remove this.
    TensorAttributes(HostStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);

    TensorAttributes(DeviceStorage storage, TensorSpec tensor_spec, TensorTopology tensor_topology);
    TensorAttributes(const TensorAttributes&) = default;
    TensorAttributes(TensorAttributes&&) = default;
    TensorAttributes& operator=(const TensorAttributes&) = default;
    TensorAttributes& operator=(TensorAttributes&&) = default;

    // Getters and setters.
    const Storage& get_storage() const;
    Storage& get_storage();
    const TensorSpec& get_tensor_spec() const;
    const TensorTopology& get_tensor_topology() const;

    TensorAttributes with_tensor_topology(TensorTopology tensor_topology) const;

private:
    Storage storage_;

    // These will be removed after Runtime Tensor refactoring,
    // as they will be part of the Host and MeshTensor,
    // Thus a part of Storage.
    TensorSpec tensor_spec_;
    TensorTopology tensor_topology_;
};

}  // namespace tt::tt_metal
