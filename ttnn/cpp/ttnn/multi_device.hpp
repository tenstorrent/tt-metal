// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "ttnn/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt_metal/impl/device/multi_device.hpp"

using Device = ttnn::Device;

namespace ttnn::multi_device {

using DeviceGrid = std::pair<int, int>;
using DeviceIds = std::vector<int>;

DeviceMesh open_device_mesh(const DeviceGrid& device_grid, const DeviceIds& device_ids, size_t l1_small_size, size_t trace_region_size, size_t num_command_queues);
void close_device_mesh(DeviceMesh &multi_device);

std::vector<ttnn::Tensor> get_device_tensors(const ttnn::Tensor& tensor);

Tensor aggregate_as_tensor(std::vector<Tensor>& tensor_shards);

}  // namespace ttnn::multi_device
