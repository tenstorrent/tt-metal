// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <variant>

#include "ttnn/distributed/distributed_tensor_config.hpp"
#include <tt-metalium/host_buffer.hpp>
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(
    Storage storage, TensorSpec tensor_spec, DistributedTensorConfig distributed_tensor_config) :
    storage_(std::move(storage)),
    tensor_spec_(std::move(tensor_spec)),
    distributed_tensor_config_(std::move(distributed_tensor_config)) {
    if (std::holds_alternative<HostStorage>(storage_)) {
        TT_FATAL(
            std::holds_alternative<ReplicateTensor>(distributed_tensor_config_),
            "Host storage is a single shard that must be in replicated configuration.");
    }
}

const Storage& TensorAttributes::get_storage() const { return storage_; }
Storage& TensorAttributes::get_storage() { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
const DistributedTensorConfig& TensorAttributes::get_distributed_tensor_config() const {
    return distributed_tensor_config_;
}

std::vector<distributed::MeshCoordinate> TensorAttributes::determine_distribution(
    const distributed::MeshShape& mesh_shape) const {
    const auto coord_range = [this, &mesh_shape]() {
        if (auto* shard2d_strategy = std::get_if<ShardTensor2D>(&distributed_tensor_config_)) {
            distributed::MeshShape distribution_shape(shard2d_strategy->shard_mesh.y, shard2d_strategy->shard_mesh.x);
            return distributed::MeshCoordinateRange(distribution_shape);
        } else {
            return distributed::MeshCoordinateRange(mesh_shape);
        }
    }();

    const int num_shards = std::visit(
        tt::stl::overloaded{
            [&mesh_shape](const HostStorage&) { return mesh_shape.mesh_size(); },
            [&mesh_shape](const DeviceStorage& s) { return s.coords.size(); },
            [&mesh_shape](const MultiDeviceHostStorage& s) { return s.num_buffers(); },
        },
        storage_);

    TT_FATAL(
        num_shards <= mesh_shape.mesh_size(),
        "Number of shards {} exceeds the mesh size {}",
        num_shards,
        mesh_shape.mesh_size());

    std::vector<distributed::MeshCoordinate> coords;
    coords.reserve(num_shards);
    auto coord_it = coord_range.begin();
    for (int i = 0; i < num_shards; ++coord_it, ++i) {
        coords.push_back(*coord_it);
    }
    return coords;
}

}  // namespace tt::tt_metal
