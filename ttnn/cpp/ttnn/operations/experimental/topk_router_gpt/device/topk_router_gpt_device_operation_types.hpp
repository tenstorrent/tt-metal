// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::topk_router_gpt {

struct operation_attributes_t {
    uint32_t k{};
    uint32_t num_experts{};
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& weight_tensor;
    const Tensor& bias_tensor;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;
using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::topk_router_gpt
