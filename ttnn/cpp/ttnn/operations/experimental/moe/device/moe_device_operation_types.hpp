// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::moe {

struct operation_attributes_t {
    MathFidelity math_fidelity = MathFidelity::LoFi;
    bool fp32_dest_acc_en = true;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& w0_tensor;
    const Tensor& w1_tensor;
    const Tensor& w2_tensor;
    const Tensor& output_tensor;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::moe
