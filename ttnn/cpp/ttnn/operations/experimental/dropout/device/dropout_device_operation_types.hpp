// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::dropout {

struct operation_attributes_t {
    const tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    const tt::tt_metal::MemoryConfig output_memory_config;

    // Specifies the seed for the dropout operation.
    // If `use_per_device_seed` is true, the seed is offset by device ID across devices in a mesh.
    uint32_t seed = 0;
    bool use_per_device_seed = false;

    const float prob = 0.0f;
    const float scale = 1.0f;
};

struct tensor_args_t {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::dropout
