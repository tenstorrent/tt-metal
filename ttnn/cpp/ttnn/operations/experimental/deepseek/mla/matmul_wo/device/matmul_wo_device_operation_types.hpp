// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::deepseek::mla {

struct operation_attributes_t {
    uint32_t layer_id{};
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& w_tensor;
    const Tensor& output_tensor;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::deepseek::mla
