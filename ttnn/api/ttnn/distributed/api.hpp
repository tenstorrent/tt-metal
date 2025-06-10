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
    const std::optional<MeshCoordinate>& offset = std::nullopt,
    const std::vector<int>& physical_device_ids = {},
    size_t worker_l1_size = DEFAULT_WORKER_L1_SIZE);

void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device);

// Given a multi-device tensor, returns a list of individual per-device tensors.
std::vector<Tensor> get_device_tensors(const Tensor& tensor);

// Given a list of per-device shards, returns multi-device tensor.
// IMPORTANT: to combine on-device shards, prefer `combine_device_tensors` instead.
// For on-host shards, a new API will be added to accommodate multi-host tensor distribution.
Tensor aggregate_as_tensor(
    const std::vector<Tensor>& tensor_shards, const tt::tt_metal::DistributedTensorConfig& config);

// Combines tensor shards allocated on individual devices into a single multi-device tensor.
// All tensors shards must be allocated on the same mesh buffer.
Tensor combine_device_tensors(const std::vector<Tensor>& tensor_shards);

std::vector<int> get_t3k_physical_device_ids_ring();

}  // namespace ttnn::distributed
