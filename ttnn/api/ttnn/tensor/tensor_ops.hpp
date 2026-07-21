// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "types.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include <tt_stl/optional_reference.hpp>
#include <ttnn/distributed/tensor_topology.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCommandQueue;
}  // namespace tt::tt_metal::distributed

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace ttnn::tensor_ops {

// Allocates a tensor on host.
// Uses `mesh_device` to allocate sufficient number of host buffers for each multi-device shard.
ttnn::Tensor allocate_tensor_on_host(
    const tt::tt_metal::TensorSpec& tensor_spec, tt::tt_metal::distributed::MeshDevice* mesh_device);
ttnn::Tensor create_device_tensor(
    const tt::tt_metal::TensorSpec& tensor_spec,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    std::optional<tt::tt_metal::TensorTopology> tensor_topology = std::nullopt);

ttnn::Tensor to_device(
    const ttnn::Tensor& input_tensor,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const tt::tt_metal::MemoryConfig> mem_config = std::nullopt,
    std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

void copy_to_device(
    const ttnn::Tensor& host_tensor,
    ttnn::Tensor& device_tensor,
    std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

void copy_to_device(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const std::byte* src,
    ttnn::Tensor& device_tensor,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt);

void copy_to_host(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const ttnn::Tensor& device_tensor,
    std::byte* dst,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt,
    bool blocking = true);

void copy_to_host(
    const ttnn::Tensor& device_tensor,
    ttnn::Tensor& host_tensor,
    bool blocking = true,
    std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

ttnn::Tensor to_layout(const ttnn::Tensor& input_tensor, tt::tt_metal::Layout target_layout);

ttnn::Tensor cpu(
    const ttnn::Tensor& input_tensor, bool blocking = true, std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

ttnn::Tensor pad(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Shape& output_padded_shape,
    const tt::tt_metal::Shape& input_tensor_start,
    float pad_value);

ttnn::Tensor unpad(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Shape& output_tensor_start,
    const tt::tt_metal::Shape& output_tensor_end);

ttnn::Tensor pad_to_tile(const ttnn::Tensor& input_tensor, float pad_value);

ttnn::Tensor unpad_from_tile(const ttnn::Tensor& input_tensor, const tt::tt_metal::Shape& output_tensor_shape);

ttnn::Tensor reshape(const ttnn::Tensor& input_tensor, const tt::tt_metal::Shape& new_shape);
ttnn::Tensor reshape(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

ttnn::Tensor view(const ttnn::Tensor& input_tensor, const tt::tt_metal::Shape& new_shape);
ttnn::Tensor view(
    const ttnn::Tensor& input_tensor,
    const tt::tt_metal::Shape& new_logical_shape,
    const tt::tt_metal::Shape& new_padded_shape);

/**
 * Reinterpret the underlying memory of input_tensor with target_layout without moving or converting data.
 *
 * The result and input_tensor will point to the same memory (whether on host or device),
 * this is a pure metadata change.
 *
 * This function is error prone, and the caller is responsible for ensuring that the reinterpretation is semantically
 * valid.
 */
ttnn::Tensor unchecked_reinterpret_layout(const ttnn::Tensor& input_tensor, tt::tt_metal::Layout target_layout);

ttnn::Tensor to_dtype(const ttnn::Tensor& input_tensor, tt::tt_metal::DataType dtype);

}  // namespace ttnn::tensor_ops

namespace ttnn {

using tensor_ops::allocate_tensor_on_host;
using tensor_ops::copy_to_device;
using tensor_ops::copy_to_host;
using tensor_ops::cpu;
using tensor_ops::create_device_tensor;
using tensor_ops::pad;
using tensor_ops::pad_to_tile;
using tensor_ops::unchecked_reinterpret_layout;
using tensor_ops::unpad;
using tensor_ops::unpad_from_tile;

}  // namespace ttnn
