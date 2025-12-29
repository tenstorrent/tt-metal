// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::moe {

struct operation_attributes_t {
    // Add any non-tensor operation attributes here
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& weight_tensor;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::moe
