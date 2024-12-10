// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <unordered_map>
#include <variant>

#include <ttnn/distributed/types.hpp>

namespace tt::tt_metal {

struct ReplicateTensor {
    int replication_factor = 1;
    ReplicateTensor() = default;
    ReplicateTensor(int replication_factor) : replication_factor(replication_factor) {}
};
bool operator==(const ReplicateTensor&, const ReplicateTensor&);
struct ShardTensor {
    int shard_dimension;
    ShardTensor(int shard_dimension) : shard_dimension(shard_dimension) {}
};
bool operator==(const ShardTensor& lhs, const ShardTensor& rhs);

struct ShardTensor2D {
    distributed::MeshShape shard_mesh;
    ShardTensor2D(distributed::MeshShape mesh) : shard_mesh(std::move(mesh)) {}
};
bool operator==(const ShardTensor2D& lhs, const ShardTensor2D& rhs);

struct AllGatherTensor {};
bool operator==(const AllGatherTensor&, const AllGatherTensor&);

// DistributedTensorConfig is a variant of different ways in which a tensor can be distributed across devices.
using DistributedTensorConfig = std::variant<ReplicateTensor, ShardTensor, ShardTensor2D, AllGatherTensor>;
DistributedTensorConfig get_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata);

}  // namespace tt::tt_metal
