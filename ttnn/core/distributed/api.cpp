// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/reflection.hpp>
#include "ttnn/distributed/api.hpp"

#include <memory>

#include <tt_stl/overloaded.hpp>
#include <tt_stl/assert.hpp>
#include <tt-metalium/distributed_host_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <ttnn/tensor/storage.hpp>
#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/host_buffer/functions.hpp>
#include <ttnn/tensor/tensor_utils.hpp>
#include <ttnn/distributed/host_ccl.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include <ttnn/distributed/types.hpp>

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

std::shared_ptr<MeshDevice> open_mesh_device(
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    const std::optional<MeshShape>& mesh_shape,
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
    if (is_cpu_tensor(tensor)) {
        std::vector<ttnn::Tensor> tensors;
        auto gathered_tensor = host_ccl::all_gather(tensor);
        const auto& distributed_buffer = gathered_tensor.host_storage().buffer();
        distributed_buffer.apply(
            [&](const HostBuffer& buffer) { tensors.push_back(Tensor{buffer, tensor.tensor_spec()}); });
        return tensors;
    }
    if (is_device_tensor(tensor) && tensor.is_allocated()) {
        const auto& device_storage = tensor.device_storage();
        std::vector<ttnn::Tensor> tensors;
        tensors.reserve(device_storage.coords.size());
        for (const auto& coord : device_storage.coords) {
            // Copies the mesh buffer, but updates the coords so the tensor only sees the single device
            DeviceStorage new_device_storage(device_storage);
            new_device_storage.coords = {coord};
            tensors.push_back(Tensor(std::move(new_device_storage), tensor.tensor_spec(), tensor.tensor_topology()));
        }
        return tensors;
    }
    return {tensor};
}

Tensor from_host_shards(const std::vector<Tensor>& tensor_shards, const MeshShape& mesh_shape, int shard_dim) {
    TT_FATAL(tensor_shards.size() == mesh_shape.mesh_size(), "Number of tensor shards must match mesh size");
    const auto& reference_shard = tensor_shards.at(0);
    for (const auto& shard : tensor_shards) {
        TT_FATAL(shard.storage_type() == StorageType::HOST, "All tensor shards must be on host");
        TT_FATAL(
            shard.tensor_spec() == reference_shard.tensor_spec(), "All tensor shards must have the same tensor spec");
    }

    auto distributed_host_buffer = DistributedHostBuffer::create(mesh_shape);
    auto shard_it = tensor_shards.begin();
    std::vector<distributed::MeshCoordinate> coords;
    for (const auto& coord : distributed::MeshCoordinateRange(mesh_shape)) {
        HostBuffer buffer = host_buffer::get_host_buffer(*(shard_it++));
        distributed_host_buffer.emplace_shard(coord, [&]() { return std::move(buffer); });
        coords.push_back(coord);
    }

    TensorTopology topology = TensorTopology::create_sharded_tensor_topology(mesh_shape, shard_dim);
    return Tensor(HostStorage{std::move(distributed_host_buffer)}, reference_shard.tensor_spec(), std::move(topology));
}

Tensor combine_device_tensors(const std::vector<Tensor>& tensor_shards, int shard_dim) {
    TT_FATAL(!tensor_shards.empty(), "At least one tensor shard must be provided");
    const auto& reference_shard = tensor_shards.at(0);
    for (const auto& shard : tensor_shards) {
        TT_FATAL(shard.storage_type() == StorageType::DEVICE, "All tensor shards must be on device");
        TT_FATAL(
            shard.tensor_spec() == reference_shard.tensor_spec(), "All tensor shards must have the same tensor spec");
    }

    std::vector<std::reference_wrapper<const DeviceStorage>> storages;
    storages.reserve(tensor_shards.size());
    for (const auto& shard : tensor_shards) {
        storages.push_back(shard.device_storage());
    }
    auto combined_storage = DeviceStorage::combine_to_multi_device_storage(storages);

    TensorTopology topology =
        TensorTopology::create_sharded_tensor_topology(MeshShape(tensor_shards.size()), shard_dim);
    return Tensor(std::move(combined_storage), reference_shard.tensor_spec(), std::move(topology));
}

}  // namespace ttnn::distributed
