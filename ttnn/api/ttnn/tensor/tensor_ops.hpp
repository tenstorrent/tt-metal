// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "types.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include <tt_stl/optional_reference.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal {
class Tensor;
class MemoryConfig;
class TensorSpec;

// Allocates a tensor on host.
// Uses `mesh_device` to allocate sufficient number of host buffers for each multi-device shard.
Tensor allocate_tensor_on_host(const TensorSpec& tensor_spec, distributed::MeshDevice* mesh_device);
Tensor create_device_tensor(const TensorSpec& tensor_spec, IDevice* device);

tt::tt_metal::Tensor to_device(
    const tt::tt_metal::Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config,
    std::optional<QueueId> cq_id);

void copy_to_device(const Tensor& host_tensor, Tensor& device_tensor, std::optional<QueueId> cq_id = std::nullopt);

Tensor to_layout(const Tensor& input_tensor, tt::tt_metal::Layout target_layout);

Tensor cpu(const Tensor& input_tensor, bool blocking, std::optional<QueueId> cq_id);

Tensor pad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

Tensor unpad(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end);

Tensor pad_to_tile(const Tensor& input_tensor, float pad_value);

Tensor unpad_from_tile(const Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape);

Tensor reshape(const Tensor& input_tensor, const tt::tt_metal::Shape& new_shape);
Tensor reshape(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

Tensor view(const Tensor& input_tensor, const tt::tt_metal::Shape& new_shape);
Tensor view(
    const Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

Tensor to_dtype(const Tensor& input_tensor, DataType dtype);

std::string to_string(const Tensor& tensor);

}  // namespace tt::tt_metal
