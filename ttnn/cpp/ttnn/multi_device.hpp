// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "tt_eager/tensor/tensor.hpp"
#include "tt_metal/impl/device/multi_device.hpp"

using Device = ttnn::Device;


namespace ttnn {

namespace multi_device {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;


inline DeviceMesh open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids) {
    return DeviceMesh(device_grid, device_ids);
}

inline void close_device_mesh(DeviceMesh &multi_device) {
    for (const auto& [device_id, device] : multi_device.managed_devices) {
        tt::tt_metal::detail::DeallocateBuffers(device.get());
        device->close();
    }
    multi_device.managed_devices.clear();
}

ttnn::Tensor to_device_mesh(const ttnn::Tensor& tensor, ttnn::multi_device::DeviceMesh& device_mesh, const ttnn::MemoryConfig& memory_config) {

    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        std::vector<DeviceBuffer> device_buffers;

        for (int i = 0; i < host_storage.buffers.size(); ++i)
        {
            Device& target_device = device_mesh.get_device(i);
            auto shard = Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()};
            shard = shard.to(&target_device, memory_config);

            device_buffers.push_back(std::get<DeviceStorage>(shard.get_storage()).buffer);
        }
        auto storage = MultiDeviceStorage{std::move(device_buffers), host_storage.shapes};
        return Tensor(std::move(storage), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage())) {
        return tensor; // already on device
    }
    TT_THROW("Expected tensor to be on MultiDeviceHostStorage type!");
}

ttnn::Tensor from_device_mesh(const ttnn::Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage())) {
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        std::vector<OwnedBuffer> host_buffers;

        for (int i = 0; i < device_storage.buffers.size(); ++i)
        {
            auto shard = Tensor{DeviceStorage{device_storage.buffers[i]}, device_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()};
            shard = shard.cpu();
            host_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
        }
        auto storage = MultiDeviceHostStorage{std::move(host_buffers), device_storage.shapes};
        return Tensor(std::move(storage), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
    }
    TT_THROW("Expected tensor to be on MultiDeviceStorage type!");
}


std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor) {
    std::vector<ttnn::Tensor> tensors;

    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage())) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.get_storage());
        for (int i = 0; i < host_storage.buffers.size(); ++i)
        {
            tensors.push_back(Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()});
        }
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage())) {
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.get_storage());
        for (int i = 0; i < device_storage.buffers.size(); ++i)
        {
            tensors.push_back(Tensor{DeviceStorage{device_storage.buffers[i]}, device_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()});
        }
        return tensors;
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
    std::vector<tt::tt_metal::Shape> shapes;
    StorageType storage_type = tensor_shards.at(0).storage_type();
    if (storage_type == StorageType::OWNED) {
        std::vector<OwnedBuffer> host_owned_buffers;
        for (const auto &shard : tensor_shards) {
            host_owned_buffers.push_back(std::get<OwnedStorage>(shard.get_storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceHostStorage{std::move(host_owned_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    } else {
        std::vector<DeviceBuffer> device_buffers;
        for (const auto &shard : tensor_shards) {
            device_buffers.push_back(std::get<DeviceStorage>(shard.get_storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceStorage{std::move(device_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    }
}

}  // namespace multi_device

}  // namespace ttnn
