// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/bcast/bcast_types.hpp"

namespace ttnn::operations::data_movement::bcast {

struct BcastParams {
    ttnn::BcastOpMath math_op;
    ttnn::BcastOpDim dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    bool in_place = false;
};

struct BcastInputs {
    Tensor input_a;
    Tensor input_b;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::operations::data_movement::bcast
