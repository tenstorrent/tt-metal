// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

enum class IntImgCB : uint32_t {
    START,                           // 8
    INPUT,                           // 8
    ACC,                             // 8
    BEFORE_ADDER_PROPAGATION_STAGE,  // 8
    OUTPUT,                          // 8
    TO_BOT_STAGE_TILE,               // 8
    FROM_TOP_STAGE_TILE,             // 8
    AXIS_3_BUFFER_0,                 // 8 (memoizing upper tile for propagation along axis 3rd/4, point of sync between
                                     // UpperWriter<->CurrentReader and CurrentReader<->CurrentCompute)
    AXIS_3_BUFFER_1,                 // 8 (dual channel)
};

struct operation_attributes_t {};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& zero_tile_tensor;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::reduction
