// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

struct operation_attributes_t {};

struct tensor_args_t {
    const Tensor& input_tensor;
};

}  // namespace ttnn::operations::experimental::reduction
