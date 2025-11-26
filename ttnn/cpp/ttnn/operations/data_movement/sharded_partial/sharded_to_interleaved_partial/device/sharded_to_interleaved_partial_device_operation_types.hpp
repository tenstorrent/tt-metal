// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation_types.hpp"

namespace ttnn::operations::data_movement {

// Partial operation specific types
struct sharded_to_interleaved_partial_operation_attributes_t {
    const uint32_t num_slices;
    const uint32_t slice_index;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const tt::tt_metal::DataType output_dtype;
};

struct sharded_to_interleaved_partial_tensor_args_t {
    const Tensor& input_tensor;
    const Tensor& cache_tensor;
};

using partial_spec_return_value_t = TensorSpec;
using partial_tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement
