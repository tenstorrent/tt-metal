// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "types.hpp"
#include "ttnn/common/queue_id.hpp"

namespace tt::tt_metal {
struct Tensor;
struct MemoryConfig;
class CommandQueue;
namespace distributed {
class MeshDevice;
}  // namespace distributed

class IDevice;

}  // namespace tt::tt_metal

namespace tt::tt_metal::tensor_ops {

Tensor tensor_to_device(
    const Tensor& input_tensor, IDevice* target_device, const MemoryConfig& mem_config, QueueId cq_id);
Tensor tensor_to_device(
    const Tensor& input_tensor, distributed::MeshDevice* mesh_device, const MemoryConfig& mem_config, QueueId cq_id);

Tensor tensor_to_layout(const Tensor& input_tensor, Layout target_layout, IDevice* worker);

Tensor tensor_cpu(const Tensor& input_tensor, bool blocking, QueueId cq_id);

void tensor_print(const Tensor& input_tensor);

Tensor tensor_pad(
    const Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value);

Tensor tensor_unpad(
    const Tensor& input_tensor, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end);

Tensor tensor_pad_to_tile(const Tensor& input_tensor, float pad_value);

Tensor tensor_unpad_from_tile(const Tensor& input_tensor, const ttnn::Shape& output_tensor_shape);

Tensor tensor_reshape(const Tensor& input_tensor, const ttnn::Shape& new_shape);
Tensor tensor_reshape(
    const Tensor& input_tensor, const ttnn::Shape& new_logical_shape, const ttnn::Shape& new_padded_shape);

}  // namespace tt::tt_metal::tensor_ops
