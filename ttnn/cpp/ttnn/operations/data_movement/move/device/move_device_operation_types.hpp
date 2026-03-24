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
// Note: output_tensor is listed first so that device_operation framework can get
// the device from it (input_tensor may be deallocated and have no device).
struct MoveTensorArgs {
    Tensor output_tensor;
    Tensor input_tensor;
};

// Return types

}  // namespace ttnn::prim
