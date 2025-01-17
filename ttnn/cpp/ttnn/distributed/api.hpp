// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/distributed_tensor_config.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed {

std::shared_ptr<MeshDevice> open_mesh_device(
    const MeshShape& mesh_shape,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    MeshType mesh_type = MeshType::RowMajor,
    const MeshOffset& offset = MeshOffset(0, 0),
    const std::vector<int>& physical_device_ids = {});

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device);

// Given a multi-device tensor, returns a list of individual per-device tensors.
std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor);

// Given a list of per-device shards, returns multi-device tensor.
Tensor aggregate_as_tensor(
    const std::vector<Tensor>& tensor_shards, const tt::tt_metal::DistributedTensorConfig& config);

std::vector<int> get_t3k_physical_device_ids_ring();

// Maps a tensor to the set of devices in the device-mesh that the shards will be distributed across.
std::vector<IDevice*> get_mapped_devices(const Tensor& tensor, MeshDevice& mesh_device);

// Get the distributed tensor config from a tensor.
tt::tt_metal::DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns the tensor on the given device.
Tensor get_device_tensor(const Tensor& multi_device_tensor, const tt::tt_metal::IDevice* device);
Tensor get_device_tensor(const Tensor& multi_device_tensor, const int device_id);

// Returns true has MultiDeviceHost/MultiDevice Storage
bool is_multi_device_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns a list of per-device tensors.
std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor);

// Given a list of per-device shards, return a multi-device tensor
Tensor create_multi_device_tensor(
    const std::vector<Tensor>& tensors,
    tt::tt_metal::StorageType storage_type,
    const tt::tt_metal::DistributedTensorConfig& strategy);

}  // namespace ttnn::distributed
