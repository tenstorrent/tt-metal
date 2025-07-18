// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/api.hpp"

#include <memory>

#include <tt_stl/overloaded.hpp>
#include "distributed/types.hpp"
#include "tt-metalium/assert.hpp"
#include "tt-metalium/distributed_host_buffer.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/storage.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>

using namespace tt::tt_metal;

namespace ttnn::distributed {

std::shared_ptr<MeshDevice> open_mesh_device(
    const MeshShape& mesh_shape,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    const std::optional<MeshCoordinate>& offset,
    const std::vector<int>& physical_device_ids,
    size_t worker_l1_size) {
    return MeshDevice::create(
        MeshDeviceConfig(mesh_shape, offset, physical_device_ids),
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config,
        {},
        worker_l1_size);
}

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device) { mesh_device->close(); }

std::vector<Tensor> get_device_tensors(const Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::HostStorage>(tensor.storage())) {
        std::vector<ttnn::Tensor> tensors;
        const auto& distributed_buffer = tensor.host_storage().buffer();
        distributed_buffer.apply(
            [&](const HostBuffer& buffer) { tensors.push_back(Tensor{buffer, tensor.get_tensor_spec()}); });
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::DeviceStorage>(tensor.storage())) {
        auto& device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.storage());
        if (auto mesh_buffer = device_storage.mesh_buffer; mesh_buffer != nullptr) {
            std::vector<ttnn::Tensor> tensors;
            tensors.reserve(device_storage.coords.size());
            for (const auto& coord : device_storage.coords) {
                DeviceStorage shard_storage(mesh_buffer, {coord});
                tensors.push_back(Tensor(
                    std::move(shard_storage), tensor.tensor_spec(), AllGatherTensor{}, tensor.tensor_topology()));
            }
            return tensors;
        } else {
            return {tensor};
        }
    } else {
        return {tensor};
    }
}

Tensor from_host_shards(const std::vector<Tensor>& tensor_shards, const MeshShape& mesh_shape) {
    TT_FATAL(tensor_shards.size() == mesh_shape.mesh_size(), "Number of tensor shards must match mesh size");
    const auto& reference_shard = tensor_shards.at(0);
    for (const auto& shard : tensor_shards) {
        TT_FATAL(shard.storage_type() == StorageType::HOST, "All tensor shards must be on host");
        TT_FATAL(
            shard.tensor_spec() == reference_shard.tensor_spec(), "All tensor shards must have the same tensor spec");
    }

    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_shape);
    auto shard_it = tensor_shards.begin();
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
        HostBuffer buffer = host_buffer::get_host_buffer(*(shard_it++));
        distributed_host_buffer.emplace_shard(coord, [&]() { return std::move(buffer); });
    }

    // TODO (#25340): Implement correct logic and add test for this
    return Tensor(
        HostStorage{std::move(distributed_host_buffer)},
        reference_shard.get_tensor_spec(),
        AllGatherTensor{},
        TensorTopology{});
}

Tensor combine_device_tensors(const std::vector<Tensor>& tensor_shards) {
    TT_FATAL(!tensor_shards.empty(), "At least one tensor shard must be provided");
    const auto& reference_shard = tensor_shards.at(0);
    for (const auto& shard : tensor_shards) {
        TT_FATAL(shard.storage_type() == StorageType::DEVICE, "All tensor shards must be on device");
        TT_FATAL(
            shard.get_tensor_spec() == reference_shard.get_tensor_spec(),
            "All tensor shards must have the same tensor spec");
    }

    auto mesh_buffer = reference_shard.device_storage().mesh_buffer;
    TT_FATAL(
        mesh_buffer != nullptr,
        "Error aggregating multichip tensors: tensors shards must be allocated on a mesh buffer.");
    std::vector<MeshCoordinate> coords;
    for (const auto& shard : tensor_shards) {
        const auto& shard_storage = shard.device_storage();
        TT_FATAL(
            shard_storage.mesh_buffer == mesh_buffer,
            "Error aggregating multichip tensors: tensor shards must be allocated on the same mesh buffer. "
            "Consider moving tensors to host, aggregating, and re-uploading on device storage.");
        for (const auto& coord : shard_storage.coords) {
            coords.push_back(coord);
        }
    }
    std::sort(coords.begin(), coords.end());
    auto duplicate =
        std::adjacent_find(coords.begin(), coords.end(), [](const auto& a, const auto& b) { return a == b; });
    TT_FATAL(duplicate == coords.end(), "Found a tensor shard at duplicate coordinate {}", *duplicate);
    // TODO (#25340): Implement correct logic and add test for this
    return Tensor(
        DeviceStorage(std::move(mesh_buffer), std::move(coords)),
        reference_shard.tensor_spec(),
        AllGatherTensor{},
        TensorTopology{});
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    using namespace tt::tt_metal::distributed;
    auto& instance = SystemMesh::instance();
    auto num_devices = instance.shape().mesh_size();
    TT_FATAL(num_devices == 8, "T3000 ring topology only works with 8 devices");

    auto physical_device_ids = instance.get_mapped_physical_device_ids(MeshShape(1, 8));
    return physical_device_ids;
}

}  // namespace ttnn::distributed
