// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::operations::data_movement {

enum class MoveOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_OVERLAP, MULTI_CORE_SHARDED };

// Operation attributes - non-tensor parameters
struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    MoveOpParallelizationStrategy move_op_parallelization_strategy;
};

// Tensor arguments - tensor parameters
struct tensor_args_t {
    Tensor input_tensor;
    Tensor output_tensor;
};

// Return types
using tensor_return_value_t = Tensor;
using shape_return_value_t = ttnn::Shape;

}  // namespace ttnn::operations::data_movement
