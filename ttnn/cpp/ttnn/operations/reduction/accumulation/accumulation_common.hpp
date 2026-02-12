// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "device/accumulation_device_operation.hpp"
#include "device/accumulation_device_operation_types.hpp"

#include "ttnn/operations/core/core.hpp"
#include <ttnn/operations/data_movement/permute/permute.hpp>
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn::operations::reduction::accumulation::common {

constexpr uint32_t FOUR_DIMENSIONS{4};
constexpr uint32_t FIRST_DIMENSION{0};

using permutation_t = ttnn::SmallVector<int64_t>;

Tensor preprocess_input_tensor(
    const Tensor& input_tensor,
    const int32_t& cum_axis,
    permutation_t& permutation,
    int32_t& accumulation_axis,
    std::optional<DataType>& dtype);

Tensor postprocess_output_tensor(
    const Tensor& output_tensor,
    const int32_t& dim,
    const permutation_t& permutation,
    const ttnn::Shape& original_shape,
    const int32_t& original_rank);

Tensor accumulation_invoke(
    const Tensor& input_tensor,
    int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_out,
    const bool& reverse_order,
    const std::optional<MemoryConfig>& memory_config,
    ttnn::prim::AccumulationOp op);

}  // namespace ttnn::operations::reduction::accumulation::common
