// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement {

struct sharded_to_interleaved_operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};
    uint32_t num_slices = 1;
    uint32_t slice_index = 0;
};

struct sharded_to_interleaved_tensor_args_t {
    Tensor input_tensor;
    std::optional<Tensor> preallocated_output;
};

using sharded_to_interleaved_spec_return_value_t = TensorSpec;
using sharded_to_interleaved_tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement
