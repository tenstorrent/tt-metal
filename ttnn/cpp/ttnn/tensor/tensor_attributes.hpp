// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/host_buffer/host_buffer.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

class TensorAttributes : public std::enable_shared_from_this<TensorAttributes> {
public:
    TensorAttributes(HostBuffer storage, TensorSpec tensor_spec);
    TensorAttributes(Storage storage, TensorSpec tensor_spec, DistributedTensorConfig strategy);

    // Getters and setters.
    const DistributedTensorConfig& get_distributed_tensor_config() const;
    const Storage& get_storage() const;
    const TensorSpec& get_tensor_spec() const;
    Storage& get_storage();

    // Determines the shards for the tensor based on the strategy and the mesh shape.
    std::vector<distributed::MeshCoordinate> determine_shards(const distributed::MeshShape& mesh_shape) const;

private:
    DistributedTensorConfig strategy_;
    Storage storage_;
    TensorSpec tensor_spec_;
};

}  // namespace tt::tt_metal
