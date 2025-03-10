// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/api.hpp"

#include <memory>

#include <tt-metalium/overloaded.hpp>
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/system_mesh.hpp>
#include "ttnn/distributed/distributed_tensor_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::distributed {

std::shared_ptr<MeshDevice> open_mesh_device(
    const MeshShape& mesh_shape,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const DispatchCoreConfig& dispatch_core_config,
    const std::optional<MeshCoordinate>& offset,
    const std::vector<int>& physical_device_ids) {
    auto config =
        MeshDeviceConfig{.mesh_shape = mesh_shape, .offset = offset, .physical_device_ids = physical_device_ids};
    return MeshDevice::create(config, l1_small_size, trace_region_size, num_command_queues, dispatch_core_config);
}

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device) { mesh_device->close(); }

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        const Tile tile = tensor.get_tensor_spec().tile();
        for (int i = 0; i < host_storage.num_buffers(); ++i) {
            tensors.push_back(Tensor{OwnedStorage{host_storage.get_buffer(i)}, host_storage.specs[i]});
        }
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        auto devices = tt::tt_metal::get_devices(tensor);
        for (auto device : devices) {
            auto shard = tt::tt_metal::get_shard_for_device(tensor, device);
            tensors.push_back(shard);
        }
        return tensors;
    } else {
        return {tensor};
    }
    TT_THROW("Expected tensor to be on MultiDeviceHostStorage type!");
}

Tensor aggregate_as_tensor(
    const std::vector<Tensor>& tensor_shards, const tt::tt_metal::DistributedTensorConfig& config) {
    TT_ASSERT(tensor_shards.size() > 0, "At least one tensor shard must be provided");
    const auto& reference_shard = tensor_shards.at(0);
    for (const auto& shard : tensor_shards) {
        if (shard.storage_type() != reference_shard.storage_type()) {
            TT_THROW("All tensor shards must have the same storage type");
        }
    }

    // Based whether the first tensor shard has OwnedBuffer or Device buffer,
    // we want to use MultiDeviceHostStorage or MultiDeviceStorage
    StorageType storage_type = reference_shard.storage_type();
    Tile tile = reference_shard.get_tensor_spec().tile();
    if (storage_type == StorageType::OWNED) {
        std::vector<ttnn::TensorSpec> specs;
        std::vector<OwnedBuffer> host_owned_buffers;
        for (const auto& shard : tensor_shards) {
            host_owned_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
            specs.push_back(shard.get_tensor_spec());
            Tile shard_tile = shard.get_tensor_spec().tile();
            if (shard_tile != tile) {
                TT_THROW(
                    "Error aggregating multichip tensors: Attempting to aggregate tensors with different tiling "
                    "configurations. Device {} has tiling ({}x{}) while device {} has tiling {}x{}.",
                    reference_shard.device()->id(),
                    tile.get_height(),
                    tile.get_width(),
                    shard.device()->id(),
                    shard_tile.get_height(),
                    shard_tile.get_width());
            }
        }
        auto storage = MultiDeviceHostStorage{config, std::move(host_owned_buffers), specs};
        return Tensor(std::move(storage), reference_shard.get_tensor_spec());
    } else {
        std::vector<int> ordered_device_ids;
        std::unordered_map<int, ttnn::TensorSpec> specs;
        std::unordered_map<int, std::shared_ptr<Buffer>> device_buffers;
        for (const auto& shard : tensor_shards) {
            IDevice* device = std::get<DeviceStorage>(shard.get_storage()).buffer->device();
            auto device_id = device->id();
            ordered_device_ids.push_back(device_id);
            device_buffers.insert({device->id(), std::get<DeviceStorage>(shard.get_storage()).buffer});
            specs.insert({device->id(), shard.get_tensor_spec()});
            Tile shard_tile = shard.get_tensor_spec().tile();
            if (shard_tile != tile) {
                TT_THROW(
                    "Error aggregating multichip tensors: Attempting to aggregate tensors with different tiling "
                    "configurations. Device {} has tiling ({}x{}) while device {} has tiling {}x{}.",
                    reference_shard.device()->id(),
                    tile.get_height(),
                    tile.get_width(),
                    shard.device()->id(),
                    shard_tile.get_height(),
                    shard_tile.get_width());
            }
        }
        auto storage =
            MultiDeviceStorage{config, ordered_device_ids, std::move(device_buffers), specs, /*mesh_buffer=*/nullptr};
        return Tensor(std::move(storage), reference_shard.get_tensor_spec());
    }
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    using namespace tt::tt_metal::distributed;
    auto& instance = SystemMesh::instance();
    auto num_devices = instance.get_shape().mesh_size();
    TT_FATAL(num_devices == 8, "T3000 ring topology only works with 8 devices");

    auto physical_device_ids = instance.get_mapped_physical_device_ids(MeshDeviceConfig{.mesh_shape = MeshShape(1, 8)});
    return physical_device_ids;
}

std::vector<IDevice*> get_mapped_devices(const Tensor& tensor, MeshDevice& mesh_device) {
    // For multi-device tensors, returns the number of workers capped by the number of buffers
    // Otherwise, returns all available workes from mesh_device.
    auto get_workers_for_tensor = [&tensor](const auto& workers) {
        if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage()) or
            std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
            return std::vector<IDevice*>(workers.begin(), workers.begin() + num_buffers_in_tensor(tensor));
        }
        return workers;
    };
    if (std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        const auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());

        // Given a MeshDevice, this linearizes the set of mapped devices for a tensor specified with some
        // distributed tensor strategy. The different strategies map to different policies on how
        // this distribution is mapped to physical devices.
        return std::visit(
            tt::stl::overloaded{
                [&](const ShardTensor2D& s) {
                    const tt::tt_metal::distributed::MeshCoordinateRange range(
                        MeshShape(s.shard_mesh.y, s.shard_mesh.x));
                    return mesh_device.get_view().get_devices(range);
                },
                [&](const auto&) { return get_workers_for_tensor(mesh_device.get_devices()); }},
            host_storage.strategy);
    } else if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        return tensor.workers;
    } else {
        return get_workers_for_tensor(mesh_device.get_devices());
    }
}

DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor) {
    if (tensor.storage_type() == StorageType::MULTI_DEVICE) {
        const auto* multi_device_storage = std::get_if<MultiDeviceStorage>(&tensor.get_storage());
        TT_ASSERT(
            multi_device_storage != nullptr,
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        return multi_device_storage->strategy;
    } else if (tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        const auto* multi_device_host_storage = std::get_if<MultiDeviceHostStorage>(&tensor.get_storage());
        TT_ASSERT(
            multi_device_host_storage != nullptr,
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        return multi_device_host_storage->strategy;
    }
    TT_THROW("Tensor is not a multi-device tensor");
}

Tensor get_device_tensor(const Tensor& multi_device_tensor, const int device_id) {
    if (const auto* tensor_storage = std::get_if<MultiDeviceStorage>(&multi_device_tensor.get_storage());
        tensor_storage != nullptr && tensor_storage->has_buffer_for_device_id(device_id)) {
        return Tensor{
            DeviceStorage{tensor_storage->get_buffer_for_device_id(device_id)},
            TensorSpec(
                multi_device_tensor.get_logical_shape(),
                TensorLayout::fromPaddedShape(
                    multi_device_tensor.get_dtype(),
                    PageConfig(multi_device_tensor.get_layout()),
                    MemoryConfig{},
                    multi_device_tensor.get_logical_shape(),
                    multi_device_tensor.get_padded_shape()))};
    } else if (std::holds_alternative<tt::tt_metal::DeviceStorage>(multi_device_tensor.get_storage())) {
        return multi_device_tensor;
    }

    TT_THROW("User is trying to access a device tensor that is not on device.");
}

Tensor get_device_tensor(const Tensor& multi_device_tensor, const IDevice* device) {
    return get_device_tensor(multi_device_tensor, device->id());
}

bool is_host_mesh_tensor(const Tensor& tensor) { return tensor.storage_type() == StorageType::MULTI_DEVICE_HOST; }

bool is_multi_device_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE or
           tensor.storage_type() == StorageType::MULTI_DEVICE_HOST;
}

bool is_mesh_buffer_tensor(const Tensor& tensor) {
    auto* multi_device_storage = std::get_if<MultiDeviceStorage>(&tensor.get_storage());
    return multi_device_storage != nullptr && multi_device_storage->mesh_buffer != nullptr;
}

std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor) {
    std::vector<ttnn::Tensor> tensors;
    if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE) {
        TT_ASSERT(
            std::holds_alternative<MultiDeviceStorage>(multi_device_tensor.get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(multi_device_tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
        tensors = std::vector<ttnn::Tensor>(tensor_storage.num_buffers(), Tensor());
        for (int i = 0; i < tensor_storage.ordered_device_ids.size(); ++i) {
            auto device_id = tensor_storage.ordered_device_ids[i];
            tensors[i] = Tensor{
                DeviceStorage{tensor_storage.get_buffer_for_device_id(device_id)}, tensor_storage.specs.at(device_id)};
        }
        return tensors;
    } else if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        TT_ASSERT(
            std::holds_alternative<MultiDeviceHostStorage>(multi_device_tensor.get_storage()),
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(multi_device_tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceHostStorage>(multi_device_tensor.get_storage());
        for (int i = 0; i < tensor_storage.num_buffers(); ++i) {
            tensors.push_back(Tensor{OwnedStorage{tensor_storage.get_buffer(i)}, tensor_storage.specs[i]});
        }
    } else {
        TT_THROW("get_tensors_from_multi_device_storage only support multi device tensors");
    }
    return tensors;
}

Tensor create_multi_device_tensor(
    const std::vector<Tensor>& tensors, StorageType storage_type, const DistributedTensorConfig& strategy) {
    if (tensors.empty()) {
        TT_THROW("Cannot create multi-device tensor with empty tensor list");
    }

    if (storage_type == StorageType::MULTI_DEVICE) {
        std::vector<int> ordered_device_ids;
        std::unordered_map<int, ttnn::TensorSpec> specs;
        std::unordered_map<int, std::shared_ptr<Buffer>> device_buffers;
        for (const auto& tensor : tensors) {
            TT_ASSERT(
                std::holds_alternative<DeviceStorage>(tensor.get_storage()),
                "Unexpected type {}",
                tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
            IDevice* device = std::get<DeviceStorage>(tensor.get_storage()).buffer->device();
            auto device_id = device->id();
            ordered_device_ids.push_back(device_id);
            device_buffers.insert({device_id, std::get<DeviceStorage>(tensor.get_storage()).buffer});
            specs.insert({device_id, tensor.get_tensor_spec()});
        }
        return Tensor{
            MultiDeviceStorage{strategy, ordered_device_ids, device_buffers, specs, /*mesh_buffer=*/nullptr},
            TensorSpec(
                tensors.at(0).get_logical_shape(),
                TensorLayout::fromPaddedShape(
                    tensors.at(0).get_dtype(),
                    PageConfig(tensors.at(0).get_layout()),
                    MemoryConfig{},
                    tensors.at(0).get_logical_shape(),
                    tensors.at(0).get_padded_shape()))};
    } else if (storage_type == StorageType::MULTI_DEVICE_HOST) {
        std::vector<OwnedBuffer> owned_buffers;
        std::vector<ttnn::TensorSpec> specs;
        for (const auto& tensor : tensors) {
            TT_ASSERT(
                std::holds_alternative<OwnedStorage>(tensor.get_storage()),
                "Unexpected type {}",
                tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
            owned_buffers.push_back(std::get<OwnedStorage>(tensor.get_storage()).buffer);
            specs.push_back(tensor.get_tensor_spec());
        }
        return Tensor{
            MultiDeviceHostStorage{strategy, owned_buffers, specs},
            TensorSpec(
                tensors.at(0).get_logical_shape(),
                TensorLayout::fromPaddedShape(
                    tensors.at(0).get_dtype(),
                    PageConfig(tensors.at(0).get_layout()),
                    MemoryConfig{},
                    tensors.at(0).get_logical_shape(),
                    tensors.at(0).get_padded_shape()))};
    } else {
        TT_THROW("Invalid storage type for multi-device tensor");
    }
}

}  // namespace ttnn::distributed
