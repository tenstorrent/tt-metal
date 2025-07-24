// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "../accumulation_common.hpp"

#include "../device/accumulation_device_operation_types.hpp"
#include "../device/accumulation_device_operation.hpp"

#include <magic_enum/magic_enum.hpp>

#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/squeeze/squeeze.hpp>
#include <ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp>

#include "tt-metalium/assert.hpp"
#include "cumprod.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor CumprodOperation::invoke(
    const QueueId& queue_id,
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config) {
    return common::accumulation_invoke(
        queue_id, input_tensor, dim, dtype, optional_out, reverse_order, memory_config, AccumulationOp::CUMPROD);
}

}  // namespace ttnn::operations::reduction::accumulation
