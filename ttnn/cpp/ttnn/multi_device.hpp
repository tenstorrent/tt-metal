// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/impl/device/device_mesh.hpp"

using Device = ttnn::Device;

namespace ttnn {
namespace multi_device {

DeviceMesh open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues, DispatchCoreType dispatch_core_type);
void close_device_mesh(DeviceMesh &multi_device);

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor);

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards);

}  // namespace multi_device

using namespace multi_device;

}  // namespace ttnn
