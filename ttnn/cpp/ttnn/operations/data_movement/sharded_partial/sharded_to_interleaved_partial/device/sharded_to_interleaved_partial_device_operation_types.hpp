// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::data_movement {

struct sharded_to_interleaved_partial_operation_attributes_t {
    uint32_t num_slices{};
    uint32_t slice_index{};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};
};

struct sharded_to_interleaved_partial_tensor_args_t {
    Tensor input_tensor;
    Tensor cache_tensor;
};

using partial_spec_return_value_t = TensorSpec;
using partial_tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement
