// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::topk_xl {

struct operation_attributes_t {
    uint32_t k{};
    bool largest{true};
    bool sorted{true};
};

struct tensor_args_t {
    const Tensor& input_tensor;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;
using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::topk_xl
