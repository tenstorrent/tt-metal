// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <variant>

#include "tt_stl/overloaded.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor_attributes.hpp"
#include "ttnn/tensor/tensor_spec.hpp"

namespace tt::tt_metal {

TensorAttributes::TensorAttributes(HostBuffer storage, TensorSpec tensor_spec) :
    storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)) {}

TensorAttributes::TensorAttributes(Storage storage, TensorSpec tensor_spec, DistributedTensorConfig strategy) :
    strategy_(std::move(strategy)), storage_(std::move(storage)), tensor_spec_(std::move(tensor_spec)) {
    TT_FATAL(
        !(std::holds_alternative<HostStorage>(storage_) && std::holds_alternative<ReplicateTensor>(strategy_)),
        "If single host storage is used, distributed tensor config must be ReplicateTensor");
}

const DistributedTensorConfig& TensorAttributes::get_distributed_tensor_config() const { return strategy_; }
const Storage& TensorAttributes::get_storage() const { return storage_; }
const TensorSpec& TensorAttributes::get_tensor_spec() const { return tensor_spec_; }
Storage& TensorAttributes::get_storage() { return storage_; }

std::vector<distributed::MeshCoordinate> TensorAttributes::determine_shards(
    const distributed::MeshShape& mesh_shape) const {
    const auto coord_range = [this, &mesh_shape]() {
        if (auto* shard2d_strategy = std::get_if<ShardTensor2D>(&strategy_)) {
            distributed::MeshShape distribution_shape(shard2d_strategy->shard_mesh.y, shard2d_strategy->shard_mesh.x);
            return distributed::MeshCoordinateRange(distribution_shape);
        } else {
            return distributed::MeshCoordinateRange(mesh_shape);
        }
    }();

    const int num_shards = std::visit(
        tt::stl::overloaded{
            [&mesh_shape](const HostStorage&) { return mesh_shape.mesh_size(); },
            [&mesh_shape](const DeviceStorage& s) { return s.shards.size(); },
            [&mesh_shape](const MultiDeviceHostStorage& s) { return s.buffers.size(); },
        },
        storage_);

    std::vector<distributed::MeshCoordinate> shards;
    shards.reserve(num_shards);
    auto coord_it = coord_range.begin();
    for (int i = 0; i < num_shards; ++coord_it, ++i) {
        shards.push_back(*coord_it);
    }
    return shards;
}

}  // namespace tt::tt_metal
