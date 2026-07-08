// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate {

struct operation_attributes_t {
    float eps{};
    float scaling_factor{};
    bool enable_sigmoid{};
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& bias_tensor;
    const Tensor& input_indices_tensor;
    const Tensor& output_tensor;
    const Tensor& output_indices_tensor;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::deepseek::moe::deepseek_moe_gate
