// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/core.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/to_dtype/to_dtype_op.hpp"
#include "ttnn/operations/core/to_layout/to_layout_op.hpp"
#include "ttnn/operations/core/to_memory_config/to_memory_config_op.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {

namespace operations {
namespace core {

ttnn::Tensor unsqueeze_to_4D(const ttnn::Tensor& tensor);

ttnn::Tensor squeeze_from_4D(const ttnn::Tensor& tensor, int rank);

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<QueueId> queue_id = std::nullopt);

ttnn::Tensor from_device(
    const ttnn::Tensor& tensor, bool blocking = true, std::optional<QueueId> queue_id = std::nullopt);

void deallocate(Tensor& tensor, bool force = true);

Tensor reallocate(const Tensor& input_tensor, const std::optional<MemoryConfig>& memory_config);

}  // namespace core
}  // namespace operations

using operations::core::deallocate;
using operations::core::from_device;
using operations::core::reallocate;
using operations::core::squeeze_from_4D;
using operations::core::to_device;
using operations::core::unsqueeze_to_4D;

constexpr auto to_dtype = ttnn::register_operation<"ttnn::to_dtype", ttnn::operations::core::ToDtype>();
constexpr auto to_memory_config =
    ttnn::register_operation<"ttnn::to_memory_config", ttnn::operations::core::ToMemoryConfig>();
constexpr auto to_layout = ttnn::register_operation<"ttnn::to_layout", ttnn::operations::core::ToLayout>();

}  // namespace ttnn
