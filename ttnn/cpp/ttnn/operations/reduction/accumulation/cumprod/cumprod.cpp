// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../accumulation_common.hpp"

#include "../device/accumulation_device_operation_types.hpp"
#include "../device/accumulation_device_operation.hpp"

#include <ttnn/operations/data_movement/permute/permute.hpp>
#include <ttnn/operations/data_movement/squeeze/squeeze.hpp>
#include <ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp>
#include <utility>

#include <tt_stl/assert.hpp>
#include "cumprod.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor CumprodOperation::invoke(
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<DataType>& dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config) {
    return common::accumulation_invoke(
        input_tensor,
        dim,
        dtype,
        std::move(optional_out),
        reverse_order,
        memory_config,
        ttnn::prim::AccumulationOp::CUMPROD);
}

}  // namespace ttnn::operations::reduction::accumulation
