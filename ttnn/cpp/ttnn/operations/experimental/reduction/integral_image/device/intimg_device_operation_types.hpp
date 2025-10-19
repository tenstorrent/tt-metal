// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

enum class IntImgCB : uint32_t {
    START,            // 2
    INPUT,            // 2
    ACC,              // 2
    CUMSUM_STAGE_0,   // 32
    CUMSUM_STAGE_1,   // 32
    CUMSUM_STAGE_2,   // 32
    CUMSUM_STAGE_3,   // 32
    OUTPUT,           // 2
    AXIS_2_BUFFER,    // 2 memoizing last tile (for the "deeper" block) for propagation along axis 2
    AXIS_3_BUFFER_0,  // 32 memoizing upper 32 tiles for propagation along axis 3
    AXIS_3_BUFFER_1,  // 32 dual channel! ^_^
};

struct operation_attributes_t {};

struct tensor_args_t {
    const Tensor& input_tensor;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::reduction
