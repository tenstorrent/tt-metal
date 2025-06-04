// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../scatter_enums.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::scatter {

struct operation_attributes_t {
    const int32_t dim;
    const tt::tt_metal::MemoryConfig output_memory_config;
    const std::optional<ScatterReductionType> opt_reduction;
};

struct tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& index_tensor;
    const Tensor& src_tensor;
    std::optional<Tensor> opt_output;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::experimental::scatter
