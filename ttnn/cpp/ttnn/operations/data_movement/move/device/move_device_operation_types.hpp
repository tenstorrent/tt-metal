// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>

namespace ttnn::prim {

enum class MoveOpParallelizationStrategy { MULTI_CORE, MULTI_CORE_OVERLAP, MULTI_CORE_SHARDED };

// Operation attributes - non-tensor parameters
struct MoveOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    MoveOpParallelizationStrategy move_op_parallelization_strategy;
    bool backwards = false;
};

// Tensor arguments - tensor parameters
struct MoveTensorArgs {
    Tensor input_tensor;
    Tensor output_tensor;
};

// Return types

}  // namespace ttnn::prim
