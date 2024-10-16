// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/distributed/types.hpp"

namespace ttnn::distributed::api {

std::shared_ptr<MeshDevice> open_mesh_device(const MeshShape& mesh_shape,
                                             size_t l1_small_size,
                                             size_t trace_region_size,
                                             size_t num_command_queues,
                                             DispatchCoreType dispatch_core_type,
                                             MeshType mesh_type = MeshType::RowMajor,
                                             const std::pair<size_t, size_t>& offset = std::pair<size_t, size_t>(0, 0),
                                             const std::vector<int>& physical_device_ids = {});

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device);

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor);

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards);

std::vector<int> get_t3k_physical_device_ids_ring();

// Maps a tensor to the set of devices in the device-mesh that the shards will be distributed across.
std::vector<Device*> distribute_tensor_to_mesh(const Tensor& tensor, MeshDevice& mesh_device);

// Get the distributed tensor config from a tensor.
DistributedTensorConfig get_distributed_tensor_config_from_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns the tensor on the given device.
Tensor get_device_tensor(const Tensor& multi_device_tensor, const Device* device);
Tensor get_device_tensor(const Tensor& multi_device_tensor, const int device_id);

// Returns true has MultiDeviceHost/MultiDevice Storage
bool is_multi_device_tensor(const Tensor& tensor);

// Given a multi-device tensor and a device, returns a list of per-device tensors.
std::vector<Tensor> get_tensors_from_multi_device_storage(const Tensor& multi_device_tensor);

// Given a list of per-device shards, return a multi-device tensor
Tensor create_multi_device_tensor(const std::vector<Tensor>& tensors,
                                  StorageType storage_type,
                                  const DistributedTensorConfig& strategy);

}  // namespace ttnn::distributed::api

namespace ttnn::distributed {

using namespace api;

}  // namespace ttnn::distributed
