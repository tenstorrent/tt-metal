// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>

namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab {

struct operation_attributes_t {
    uint32_t layer_id{};
    uint32_t pos{};
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& w_a_tensor;
    const Tensor& wq_b_tensor;
    const Tensor& rope_tensor;
    const Tensor& output_tensor;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::deepseek::mla::mla_wqkv_ab
