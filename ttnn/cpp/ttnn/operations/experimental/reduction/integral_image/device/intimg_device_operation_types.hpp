// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

enum class IntImgCB : uint32_t {
    START,
    INPUT,
    ACC,
    CUMSUM_STAGE_0,
    CUMSUM_STAGE_1,
    CUMSUM_STAGE_2,
    CUMSUM_STAGE_3,
    OUTPUT,
    AXIS_2_BUFFER,    // memoizing last tile (for the "deeper" block) for propagation along axis 2
    AXIS_3_BUFFER_0,  // memoizing upper 32 tiles for propagation along axis 3
    AXIS_3_BUFFER_1,  // dual channel
};

struct operation_attributes_t {};

struct tensor_args_t {
    const Tensor& input_tensor;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::reduction
