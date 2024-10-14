// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/api.hpp"

#include <memory>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tt_metal/distributed/mesh_device.hpp"

namespace ttnn::distributed::api {

std::shared_ptr<MeshDevice> open_mesh_device(const MeshShape& mesh_shape, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues, DispatchCoreType dispatch_core_type, MeshType mesh_type, const std::pair<size_t, size_t>& offset, const std::vector<int>& physical_device_ids) {
    auto config = MeshDeviceConfig(mesh_shape, offset, physical_device_ids, mesh_type);
    return MeshDevice::create(config, l1_small_size, trace_region_size, num_command_queues, dispatch_core_type);
}

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device) {
    mesh_device->close_devices();
}

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        for (int i = 0; i < host_storage.num_buffers(); ++i)
        {
            tensors.push_back(Tensor{OwnedStorage{host_storage.get_buffer(i)},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()});
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

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards)
{
    TT_ASSERT(tensor_shards.size() > 0, "At least one tensor shard must be provided");
    for (const auto &shard : tensor_shards) {
        if (shard.storage_type() != tensor_shards.at(0).storage_type()) {
            TT_THROW("All tensor shards must have the same storage type");
        }
    }

    // Based whether the first tensor shard has OwnedBuffer or Device buffer,
    // we want to use MultiDeviceHostStorage or MultiDeviceStorage
    StorageType storage_type = tensor_shards.at(0).storage_type();
    if (storage_type == StorageType::OWNED) {
        std::vector<tt::tt_metal::LegacyShape> shapes;
        std::vector<OwnedBuffer> host_owned_buffers;
        for (const auto &shard : tensor_shards) {
            host_owned_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceHostStorage{AllGatherTensor(), std::move(host_owned_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    } else {
        std::vector<int> ordered_device_ids;
        std::unordered_map<int, tt::tt_metal::LegacyShape> shapes;
        std::unordered_map<int, DeviceBuffer> device_buffers;
        for (const auto &shard : tensor_shards) {
            Device* device = std::get<DeviceStorage>(shard.get_storage()).buffer->device();
            auto device_id = device->id();
            ordered_device_ids.push_back(device_id);
            device_buffers.insert({device->id(), std::get<DeviceStorage>(shard.get_storage()).buffer});
            shapes.insert({device->id(), shard.get_legacy_shape()});
        }
        auto storage = MultiDeviceStorage{AllGatherTensor(), ordered_device_ids, std::move(device_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    }
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    using namespace tt::tt_metal::distributed;
    auto& instance = SystemMesh::instance();
    auto num_devices = instance.get_num_devices();
    TT_FATAL(num_devices == 8, "T3000 ring topology only works with 8 devices");

    auto physical_device_ids = instance.get_mapped_physical_device_ids(
        MeshDeviceConfig(MeshShape{1, 8}, MeshOffset{0, 0}));
    return physical_device_ids;
}

std::vector<Device*> distribute_tensor_to_mesh(const Tensor& tensor, MeshDevice& mesh_device) {
    auto get_multi_device_workers = [&](const std::vector<Device*>& workers) {
        if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage()) or
            std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
            return std::vector<Device*>(workers.begin(), workers.begin() + num_buffers_in_tensor(tensor));
        }
        return workers;
    };

    if (mesh_device.get_view() != nullptr and std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
        const auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());

        return std::visit([&](const auto& strategy) {
            using StrategyType = std::decay_t<decltype(strategy)>;
            if constexpr (std::is_same_v<StrategyType, ShardTensor2D>) {
                auto mesh_view = mesh_device.get_view();
                return mesh_view->get_devices(strategy.shard_mesh);
            } else {
                return get_multi_device_workers(mesh_device.get_devices());
            }
        }, host_storage.strategy);
    } else if (std::holds_alternative<MultiDeviceStorage>(tensor.get_storage())) {
        return tensor.workers;
    } else {
        return get_multi_device_workers(mesh_device.get_devices());
    }
}

DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor) {
    if (tensor.storage_type() == StorageType::MULTI_DEVICE) {
        TT_ASSERT(std::holds_alternative<MultiDeviceStorage>(tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceStorage>(tensor.get_storage());
        return tensor_storage.strategy;
    } else if (tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        TT_ASSERT(std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceHostStorage>(tensor.get_storage());
        return tensor_storage.strategy;
    }
    TT_THROW("Tensor is not a multi-device tensor");
}

Tensor get_device_tensor(const Tensor& multi_device_tensor, const int device_id) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(multi_device_tensor.get_storage())) {
        const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
        if (tensor_storage.has_buffer_for_device_id(device_id)) {
            return Tensor{
                DeviceStorage{tensor_storage.get_buffer_for_device_id(device_id)},
                multi_device_tensor.get_legacy_shape(),
                multi_device_tensor.get_dtype(),
                multi_device_tensor.get_layout()};
        }
    } else if (std::holds_alternative<tt::tt_metal::DeviceStorage>(multi_device_tensor.get_storage())) {
        return multi_device_tensor;
    }

    TT_THROW("User is trying to access a device tensor that is not on device.");
}

Tensor get_device_tensor(const Tensor& multi_device_tensor, const Device* device) {
    return get_device_tensor(multi_device_tensor, device->id());
}

bool is_multi_device_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE or
           tensor.storage_type() == StorageType::MULTI_DEVICE_HOST;
}

std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor) {
    std::vector<ttnn::Tensor> tensors;
    if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE) {
        TT_ASSERT(std::holds_alternative<MultiDeviceStorage>(multi_device_tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(multi_device_tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceStorage>(multi_device_tensor.get_storage());
        tensors = std::vector<ttnn::Tensor>(tensor_storage.num_buffers(), Tensor());
        for (int i = 0; i < tensor_storage.ordered_device_ids.size(); ++i) {
            auto device_id = tensor_storage.ordered_device_ids[i];
            tensors[i] = Tensor{
                DeviceStorage{tensor_storage.get_buffer_for_device_id(device_id)},
                tensor_storage.shapes.at(device_id),
                multi_device_tensor.get_dtype(),
                multi_device_tensor.get_layout()};
        }
        return tensors;
    } else if (multi_device_tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        TT_ASSERT(std::holds_alternative<MultiDeviceHostStorage>(multi_device_tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(multi_device_tensor.get_storage()));
        const auto& tensor_storage = std::get<MultiDeviceHostStorage>(multi_device_tensor.get_storage());
        for (int i = 0; i < tensor_storage.num_buffers(); ++i) {
            tensors.push_back(Tensor{
                OwnedStorage{tensor_storage.get_buffer(i)},
                tensor_storage.shapes[i],
                multi_device_tensor.get_dtype(),
                multi_device_tensor.get_layout()});
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
        std::unordered_map<int, tt::tt_metal::LegacyShape> shapes;
        std::unordered_map<int, DeviceBuffer> device_buffers;
        for (const auto& tensor : tensors) {
            TT_ASSERT(std::holds_alternative<DeviceStorage>(tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
            Device* device = std::get<DeviceStorage>(tensor.get_storage()).buffer->device();
            auto device_id = device->id();
            ordered_device_ids.push_back(device_id);
            device_buffers.insert({device_id, std::get<DeviceStorage>(tensor.get_storage()).buffer});
            shapes.insert({device_id, tensor.get_legacy_shape()});
        }
        return Tensor{
            MultiDeviceStorage{strategy, ordered_device_ids, device_buffers, shapes},
            tensors.at(0).get_legacy_shape(),
            tensors.at(0).get_dtype(),
            tensors.at(0).get_layout()};
    } else if (storage_type == StorageType::MULTI_DEVICE_HOST) {
        std::vector<OwnedBuffer> owned_buffers;
        std::vector<tt::tt_metal::LegacyShape> shapes;
        for (const auto& tensor : tensors) {
            TT_ASSERT(std::holds_alternative<OwnedStorage>(tensor.get_storage()), "Unexpected type {}", tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
            owned_buffers.push_back(std::get<OwnedStorage>(tensor.get_storage()).buffer);
            shapes.push_back(tensor.get_legacy_shape());
        }
        return Tensor{
            MultiDeviceHostStorage{strategy, owned_buffers, shapes},
            tensors.at(0).get_legacy_shape(),
            tensors.at(0).get_dtype(),
            tensors.at(0).get_layout()};
    } else {
        TT_THROW("Invalid storage type for multi-device tensor");
    }
}

}  // namespace ttnn::distributed
