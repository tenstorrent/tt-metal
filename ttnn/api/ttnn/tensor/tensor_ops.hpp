// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "types.hpp"

#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/layout/layout.hpp"
#include <tt-metalium/tile.hpp>
#include <tt_stl/optional_reference.hpp>
#include <ttnn/distributed/tensor_topology.hpp>
#include "ttnn/tensor/tensor_spec.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshCommandQueue;
}  // namespace tt::tt_metal::distributed

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace ttnn {

// Allocates a tensor on host.
// Uses `mesh_device` to allocate sufficient number of host buffers for each multi-device shard.
Tensor allocate_tensor_on_host(
    const tt::tt_metal::TensorSpec& tensor_spec, tt::tt_metal::distributed::MeshDevice* mesh_device);
Tensor create_device_tensor(
    const tt::tt_metal::TensorSpec& tensor_spec,
    tt::tt_metal::distributed::MeshDevice* mesh_device,
    std::optional<tt::tt_metal::TensorTopology> tensor_topology = std::nullopt);

void copy_to_device(
    const Tensor& host_tensor, Tensor& device_tensor, std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

void copy_to_device(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const std::byte* src,
    Tensor& device_tensor,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt);

void copy_to_host(
    tt::tt_metal::distributed::MeshCommandQueue& queue,
    const Tensor& device_tensor,
    std::byte* dst,
    const std::optional<tt::tt_metal::BufferRegion>& region = std::nullopt,
    bool blocking = true);

void copy_to_host(
    const Tensor& device_tensor,
    Tensor& host_tensor,
    bool blocking = true,
    std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

Tensor cpu(const Tensor& input_tensor, bool blocking = true, std::optional<tt::tt_metal::QueueId> cq_id = std::nullopt);

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

/**
 * Reinterpret the underlying memory of input_tensor with target_layout without moving or converting data.
 *
 * The result and input_tensor will point to the same memory (whether on host or device),
 * this is a pure metadata change.
 *
 * This function is error prone, and the caller is responsible for ensuring that the reinterpretation is semantically
 * valid.
 */
Tensor unchecked_reinterpret_layout(const Tensor& input_tensor, tt::tt_metal::Layout target_layout);

Tensor to_dtype(const Tensor& input_tensor, const tt::tt_metal::DataType& dtype);

}  // namespace ttnn

// These will be moved to ttnn namespace, they are here as ttnn functions with the same name already exist.
namespace tt::tt_metal {

ttnn::Tensor to_device(
    const ttnn::Tensor& input_tensor,
    distributed::MeshDevice* mesh_device,
    ttsl::optional_reference<const MemoryConfig> mem_config = std::nullopt,
    std::optional<QueueId> cq_id = std::nullopt);

ttnn::Tensor to_layout(
    const ttnn::Tensor& input_tensor, Layout target_layout, ttsl::optional_reference<const Tile> tile = std::nullopt);

ttnn::Tensor view(const ttnn::Tensor& input_tensor, const Shape& new_shape);
ttnn::Tensor view(const ttnn::Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape);

ttnn::Tensor reshape(const ttnn::Tensor& input_tensor, const Shape& new_shape);
ttnn::Tensor reshape(const ttnn::Tensor& input_tensor, const Shape& new_logical_shape, const Shape& new_padded_shape);

}  // namespace tt::tt_metal
