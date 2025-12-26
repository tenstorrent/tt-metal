// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "kernels/op_types.hpp"

namespace ttnn::operations::reduction {

struct operation_attributes_t {
    const std::optional<MemoryConfig> output_memory_config;
    const std::optional<DataType> output_dtype;
    const tt::tt_metal::MemoryConfig output_mem_config;
    OpType op_type{OpType::IDENTITY};
    float scaler{1.0f};
};

struct tensor_args_t {
    const Tensor& input;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::reduction
