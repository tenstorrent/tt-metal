// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/distributed/api.hpp"

#include <memory>

#include <tt_stl/overloaded.hpp>
#include "tt-metalium/assert.hpp"
#include "tt-metalium/mesh_coord.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
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
    return MeshDevice::create(
        MeshDeviceConfig(mesh_shape, offset, physical_device_ids),
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config);
}

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device) { mesh_device->close(); }

std::vector<Tensor> get_device_tensors(const Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        std::vector<ttnn::Tensor> tensors;
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        const Tile tile = tensor.get_tensor_spec().tile();
        for (int i = 0; i < host_storage.num_buffers(); ++i) {
            tensors.push_back(Tensor{OwnedStorage{host_storage.get_buffer(i)}, host_storage.specs[i]});
        }
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::DeviceStorage>(tensor.get_storage())) {
        auto& device_storage = std::get<tt::tt_metal::DeviceStorage>(tensor.get_storage());
        if (auto mesh_buffer = device_storage.mesh_buffer; mesh_buffer != nullptr) {
            std::vector<ttnn::Tensor> tensors;
            tensors.reserve(device_storage.specs.size());
            for (const auto& [coord, shard_spec] : device_storage.specs) {
                DeviceStorage shard_storage(mesh_buffer, AllGatherTensor{}, {std::make_pair(coord, shard_spec)});
                tensors.push_back(Tensor(std::move(shard_storage), shard_spec));
            }
            return tensors;
        } else {
            return {tensor};
        }
    } else {
        return {tensor};
    }
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
    } else if (storage_type == StorageType::BORROWED) {
        std::vector<ttnn::TensorSpec> specs;
        std::vector<OwnedBuffer> host_owned_buffers;
        for (const auto& shard : tensor_shards) {
            auto buffer = std::get<BorrowedStorage>(shard.get_storage()).buffer;
            specs.push_back(shard.get_tensor_spec());

            auto visitor = tt::stl::overloaded{[&shard, &host_owned_buffers](const auto& buffer) -> OwnedBuffer {
                using BorrowedBufferType = std::vector<typename std::decay_t<decltype(buffer)>::value_type>;

                return owned_buffer::create(BorrowedBufferType(buffer.begin(), buffer.end()));
            }};

            host_owned_buffers.push_back(std::visit(visitor, buffer));
        }
        auto storage = MultiDeviceHostStorage{config, std::move(host_owned_buffers), specs};
        return Tensor(std::move(storage), reference_shard.get_tensor_spec());
    } else if (storage_type == StorageType::DEVICE) {
        auto mesh_buffer = std::get<DeviceStorage>(reference_shard.get_storage()).mesh_buffer;
        TT_FATAL(
            mesh_buffer != nullptr,
            "Error aggregating multichip tensors: tensors shards must be allocated on a mesh buffer.");
        std::vector<std::pair<MeshCoordinate, TensorSpec>> specs;

        for (const auto& shard : tensor_shards) {
            const auto& shard_storage = std::get<DeviceStorage>(shard.get_storage());
            TT_FATAL(
                shard_storage.mesh_buffer == mesh_buffer,
                "Error aggregating multichip tensors: tensor shards must be allocated on the same mesh buffer. "
                "Consider moving tensors to host, aggregating, and re-uploading on device storage.");
            for (const auto& [coord, shard_spec] : shard_storage.specs) {
                specs.push_back(std::make_pair(coord, shard_spec));
            }
        }
        std::sort(specs.begin(), specs.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        auto duplicate = std::adjacent_find(
            specs.begin(), specs.end(), [](const auto& a, const auto& b) { return a.first == b.first; });
        TT_FATAL(duplicate == specs.end(), "Found a tensor shard at duplicate coordiante {0}", duplicate->first);

        auto storage = DeviceStorage(mesh_buffer, AllGatherTensor{}, specs);
        return Tensor(std::move(storage), reference_shard.get_tensor_spec());
    } else {
        TT_THROW(
            "Unsupported storage type for multi-device tensor: {}",
            tt::stl::get_active_type_name_in_variant(reference_shard.get_storage()));
    }
}

std::vector<int> get_t3k_physical_device_ids_ring() {
    using namespace tt::tt_metal::distributed;
    auto& instance = SystemMesh::instance();
    auto num_devices = instance.get_shape().mesh_size();
    TT_FATAL(num_devices == 8, "T3000 ring topology only works with 8 devices");

    auto physical_device_ids = instance.get_mapped_physical_device_ids(MeshShape(1, 8));
    return physical_device_ids;
}

std::vector<IDevice*> get_mapped_devices(const Tensor& tensor, MeshDevice& mesh_device) {
    // For multi-device tensors, returns the number of workers capped by the number of buffers
    // Otherwise, returns all available workes from mesh_device.
    auto get_workers_for_tensor = [&tensor](const auto& workers) {
        if (std::holds_alternative<MultiDeviceHostStorage>(tensor.get_storage())) {
            const auto num_buffers = std::get<MultiDeviceHostStorage>(tensor.get_storage()).num_buffers();
            return std::vector<IDevice*>(workers.begin(), workers.begin() + num_buffers);
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
    } else {
        return get_workers_for_tensor(mesh_device.get_devices());
    }
}

DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor) {
    if (tensor.storage_type() == StorageType::MULTI_DEVICE_HOST) {
        const auto* multi_device_host_storage = std::get_if<MultiDeviceHostStorage>(&tensor.get_storage());
        TT_ASSERT(
            multi_device_host_storage != nullptr,
            "Unexpected type {}",
            tt::stl::get_active_type_name_in_variant(tensor.get_storage()));
        return multi_device_host_storage->strategy;
    } else if (tensor.storage_type() == StorageType::DEVICE) {
        const auto& device_storage = std::get<DeviceStorage>(tensor.get_storage());
        TT_FATAL(device_storage.mesh_buffer != nullptr, "Device storage must be on a mesh buffer");
        return device_storage.strategy;
    } else {
        TT_THROW("Tensor is not a multi-device tensor");
    }
}

bool is_multi_device_host_tensor(const Tensor& tensor) {
    return tensor.storage_type() == StorageType::MULTI_DEVICE_HOST;
}

bool is_mesh_buffer_tensor(const Tensor& tensor) {
    const auto* device_storage = std::get_if<DeviceStorage>(&tensor.get_storage());
    return device_storage != nullptr && device_storage->mesh_buffer != nullptr;
}

}  // namespace ttnn::distributed
