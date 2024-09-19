// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/impl/device/mesh_device.hpp"

using Device = ttnn::Device;

namespace ttnn {
namespace multi_device {

std::shared_ptr<MeshDevice> open_mesh_device(const MeshShape& mesh_shape, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues, DispatchCoreType dispatch_core_type, const std::pair<size_t, size_t>& offset = {0, 0});
void close_mesh_device(const std::shared_ptr<MeshDevice>& mesh_device);

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor);

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards);

std::vector<int> get_t3k_physical_device_ids_ring();

}  // namespace multi_device

using namespace multi_device;

}  // namespace ttnn
