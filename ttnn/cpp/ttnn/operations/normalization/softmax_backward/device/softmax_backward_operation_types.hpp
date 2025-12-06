// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <cstdint>

namespace ttnn::operations::normalization::softmax_backward {

struct operation_attributes_t {
    const uint32_t dim;
};

struct tensor_args_t {
    const ttnn::Tensor& softmax_output;
    const ttnn::Tensor& upstream_grad;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttnn::operations::normalization::softmax_backward
