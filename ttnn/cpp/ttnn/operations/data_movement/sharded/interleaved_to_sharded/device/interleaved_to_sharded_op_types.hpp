// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement::interleaved_to_sharded {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool keep_l1_aligned;
};

struct tensor_args_t {
    tt::tt_metal::Tensor input_tensor;
    std::optional<tt::tt_metal::Tensor> preallocated_output;
};

using spec_return_value_t = tt::tt_metal::TensorSpec;
using tensor_return_value_t = tt::tt_metal::Tensor;

}  // namespace ttnn::operations::data_movement::interleaved_to_sharded
