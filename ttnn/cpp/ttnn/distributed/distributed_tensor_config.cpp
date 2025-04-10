// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <unordered_map>
#include <string>
#include <magic_enum/magic_enum.hpp>
#include <algorithm>  // Required for std::transform
#include <cctype>     // Required for ::toupper

#include "ttnn/distributed/distributed_tensor_config.hpp"

namespace tt::tt_metal {
namespace {

DistributedTensorConfig create_shard_distributed_tensor_config(
    const std::unordered_map<std::string, std::string>& metadata) {
    std::string linearization_str = metadata.at("linearization");
    std::transform(linearization_str.begin(), linearization_str.end(), linearization_str.begin(), ::toupper);
    auto linearization_type = magic_enum::enum_cast<DeviceLinearizationType>(linearization_str);
    TT_FATAL(linearization_type.has_value(), "Unsupported DeviceLinearizationType:", metadata.at("linearization"));
    return ShardTensor(std::stoi(metadata.at("shard_dim")), linearization_type.value());
}

DistributedTensorConfig create_shard_2d_distributed_tensor_config(
    const std::unordered_map<std::string, std::string>& metadata) {
    return ShardTensor2D(ShardMesh(std::stoi(metadata.at("mesh_shape_y")), std::stoi(metadata.at("mesh_shape_x"))));
}

DistributedTensorConfig create_replicate_distributed_tensor_config(
    const std::unordered_map<std::string, std::string>& metadata) {
    if (auto it = metadata.find("replication_factor"); it != metadata.end()) {
        return ReplicateTensor(std::stoi(it->second));
    }
    TT_THROW("Unsupported Replication strategy:");
}
}  // namespace

DistributedTensorConfig get_distributed_tensor_config(const std::unordered_map<std::string, std::string>& metadata) {
    if (auto it = metadata.find("strategy"); it != metadata.end()) {
        const std::string& strategy = it->second;
        if (strategy == "shard") {
            return create_shard_distributed_tensor_config(metadata);
        } else if (strategy == "shard_2d") {
            return create_shard_2d_distributed_tensor_config(metadata);
        } else if (strategy == "replicate") {
            return create_replicate_distributed_tensor_config(metadata);
        }
    }
    TT_THROW("Unsupported DistributedTensorConfig strategy:");
}

bool operator==(const ReplicateTensor& a, const ReplicateTensor& b) {
    return a.replication_factor == b.replication_factor;
}
bool operator==(const AllGatherTensor&, const AllGatherTensor&) {
    // All instances are considered equal because there are no data members.
    return true;
}
bool operator==(const ShardTensor& lhs, const ShardTensor& rhs) {
    return lhs.shard_dimension == rhs.shard_dimension && lhs.linearization == rhs.linearization;
}
bool operator==(const ShardTensor2D& lhs, const ShardTensor2D& rhs) {
    return lhs.shard_mesh.x == rhs.shard_mesh.x &&  //
           lhs.shard_mesh.y == rhs.shard_mesh.y;
}

}  // namespace tt::tt_metal
