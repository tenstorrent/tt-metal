// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {

namespace operations {
namespace core {

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor);

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, const int rank);

ttnn::Tensor to_device(const ttnn::Tensor& tensor, Device* device, const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor, MeshDevice* mesh_device, const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape,
    DataType data_type,
    Layout layout,
    Device* device,
    const std::optional<MemoryConfig>& memory_config);

ttnn::Tensor allocate_tensor_on_device(
    const Shape& shape,
    DataType data_type,
    Layout layout,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config);

void copy_host_to_device_tensor(ttnn::Tensor host_tensor, ttnn::Tensor device_tensor, uint8_t cq_id = ttnn::DefaultQueueId);

ttnn::Tensor from_device(const ttnn::Tensor& tensor, bool blocking = true, uint8_t cq_id = ttnn::DefaultQueueId);

void deallocate(Tensor& tensor, bool force = true);

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);

// Trace APIs - Single Device
uint32_t begin_trace_capture(Device* device, const uint8_t cq_id);

void end_trace_capture(Device* device, const uint32_t tid, const uint8_t cq_id);

void execute_trace(Device* device, const uint32_t tid, const uint8_t cq_id, bool blocking);

void release_trace(Device* device, const uint32_t tid);

// Trace APIs - Multi Device
uint32_t begin_trace_capture(MeshDevice* device, const uint8_t cq_id = ttnn::DefaultQueueId);

void end_trace_capture(MeshDevice* device, const uint32_t tid, const uint8_t cq_id = ttnn::DefaultQueueId);

void execute_trace(MeshDevice* device, const uint32_t tid, const uint8_t cq_id = ttnn::DefaultQueueId, bool blocking = true);

void release_trace(MeshDevice* device, const uint32_t tid);

}  // namespace core
}  // namespace operations

using operations::core::deallocate;
using operations::core::from_device;
using operations::core::reallocate;
using operations::core::squeeze_from_4D;
using operations::core::to_device;
using operations::core::unsqueeze_to_4D;

constexpr auto to_dtype = ttnn::register_operation_with_auto_launch_op<"ttnn::to_dtype", ttnn::operations::core::ToDtype>();
constexpr auto to_memory_config =
    ttnn::register_operation_with_auto_launch_op<"ttnn::to_memory_config", ttnn::operations::core::ToMemoryConfig>();
constexpr auto to_layout = ttnn::register_operation_with_auto_launch_op<"ttnn::to_layout", ttnn::operations::core::ToLayout>();

}  // namespace ttnn
