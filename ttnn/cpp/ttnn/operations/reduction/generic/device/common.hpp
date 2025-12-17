
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal {

enum class ReduceOpMath : std::uint8_t { SUM, MAX, MIN };

enum class ReduceOpDim : std::uint8_t { H, W, HW };

enum class ReduceOpParallelizationStrategy : std::uint8_t { MULTI_CORE_H, MULTI_CORE_W, MULTI_CORE_HW, SINGLE_CORE_HW };

}  // namespace tt::tt_metal

namespace ttnn::operations::reduction::generic::detail {

tt::tt_metal::ReduceOpParallelizationStrategy get_parallelization_strategy(
    const tt::tt_metal::Tensor& input_tensors, tt::tt_metal::ReduceOpDim reduce_dim);

}  // namespace ttnn::operations::reduction::generic::detail
