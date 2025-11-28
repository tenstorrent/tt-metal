// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::data_movement {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
    int64_t dim;
};

struct tensor_args_t {
    Tensor batch_id;
    Tensor input_tensor_a;
    Tensor input_tensor_b;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::data_movement
