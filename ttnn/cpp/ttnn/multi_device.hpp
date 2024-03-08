// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_pool.hpp"
#include "ttnn/device.hpp"
#include "ttnn/types.hpp"
#include "tt_eager/tensor/tensor.hpp"

namespace ttnn {

namespace multi_device {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

class DeviceMesh
{
public:
    DeviceGrid device_grid;
    DeviceIds device_ids;
    std::vector<Device*> managed_devices;

    DeviceMesh(const DeviceGrid& device_grid, const DeviceIds &device_ids)
        : device_grid(device_grid), device_ids(device_ids)
    {
        auto num_requested_devices = device_ids.size();
        auto num_available_devices = tt::tt_metal::GetNumAvailableDevices();

        managed_devices.resize(num_requested_devices, nullptr);
        for (int i = 0; i < num_requested_devices; i++) { // assume linear ordering
            auto device_id = device_ids[i];
            TT_ASSERT(device_id < num_available_devices);
            if (managed_devices[i] == nullptr) {
                managed_devices[i] = &ttnn::device::open_device(device_id);
            }
        }
    }
    ~DeviceMesh() = default;

    DeviceMesh(const DeviceMesh &) = delete;
    DeviceMesh &operator=(const DeviceMesh &) = delete;

    DeviceMesh(DeviceMesh &&) = delete;
    DeviceMesh &operator=(DeviceMesh &&) = delete;

    Device &get_device(int index)
    {
        for (int i = 0; i < managed_devices.size(); i++) {
            if (device_ids[i] == index) {
                return *managed_devices[i];
            }
        }
        TT_THROW("User has provided an invalid device index");
    }

    const DeviceIds &get_device_ids() const
    {
        return device_ids;
    }

    int num_devices() const
    {
        return managed_devices.size();
    }
};

std::unordered_map<int, std::shared_ptr<DeviceMesh>> id_to_multi_device;

inline DeviceMesh &open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids) {
    auto multi_device = std::make_shared<DeviceMesh>(device_grid, device_ids);
    for (auto device_id : device_ids) {
        id_to_multi_device[device_id] = multi_device;
    }
    return *multi_device;
}

inline void close_device_mesh(DeviceMesh &multi_device) {
    for (int i = 0; i < multi_device.managed_devices.size(); i++) {
        id_to_multi_device.erase(multi_device.managed_devices[i]->id());
        ttnn::device::close_device(*multi_device.managed_devices[i]);
        multi_device.managed_devices[i] = nullptr;
    }
}

ttnn::Tensor to_device_mesh(const ttnn::Tensor& tensor, ttnn::multi_device::DeviceMesh& device_mesh, const ttnn::MemoryConfig& memory_config) {

    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.storage())) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.storage());
        std::vector<DeviceBuffer> device_buffers;

        for (int i = 0; i < host_storage.buffers.size(); ++i)
        {
            Device& target_device = device_mesh.get_device(i);
            auto shard = Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()};
            shard = shard.to(&target_device);

            device_buffers.push_back(std::get<DeviceStorage>(shard.storage()).buffer);
        }
        auto storage = MultiDeviceStorage{std::move(device_buffers), host_storage.shapes};
        return Tensor(std::move(storage), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
    }
    TT_THROW("Expected tensor to be on MultiDeviceHostStorage type!");
}

ttnn::Tensor from_device_mesh(const ttnn::Tensor& tensor) {
    if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.storage())) {
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.storage());
        std::vector<OwnedBuffer> host_buffers;

        for (int i = 0; i < device_storage.buffers.size(); ++i)
        {
            auto shard = Tensor{DeviceStorage{device_storage.buffers[i]}, device_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()};
            shard = shard.cpu();
            host_buffers.push_back(std::get<OwnedStorage>(shard.storage()).buffer);
        }
        auto storage = MultiDeviceHostStorage{std::move(host_buffers), device_storage.shapes};
        return Tensor(std::move(storage), tensor.get_legacy_shape(), tensor.get_dtype(), tensor.get_layout());
    }
    TT_THROW("Expected tensor to be on MultiDeviceStorage type!");
}


std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor) {
    std::vector<ttnn::Tensor> tensors;

    if (std::holds_alternative<tt::tt_metal::MultiDeviceHostStorage>(tensor.storage())) {
        auto& host_storage = std::get<tt::tt_metal::MultiDeviceHostStorage>(tensor.storage());
        for (int i = 0; i < host_storage.buffers.size(); ++i)
        {
            tensors.push_back(Tensor{OwnedStorage{host_storage.buffers[i]},  host_storage.shapes[i], tensor.get_dtype(), tensor.get_layout()});
        }
        return tensors;
    } else if (std::holds_alternative<tt::tt_metal::MultiDeviceStorage>(tensor.storage())) {
        auto& device_storage = std::get<tt::tt_metal::MultiDeviceStorage>(tensor.storage());
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
            host_owned_buffers.push_back(std::get<OwnedStorage>(shard.storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceHostStorage{std::move(host_owned_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    } else {
        std::vector<DeviceBuffer> device_buffers;
        for (const auto &shard : tensor_shards) {
            device_buffers.push_back(std::get<DeviceStorage>(shard.storage()).buffer);
            shapes.push_back(shard.get_legacy_shape());
        }
        auto storage = MultiDeviceStorage{std::move(device_buffers), shapes};
        return Tensor(std::move(storage), tensor_shards.at(0).get_legacy_shape(), tensor_shards.at(0).get_dtype(),  tensor_shards.at(0).get_layout());
    }
}

}  // namespace multi_device

}  // namespace ttnn
