// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads {

struct operation_attributes_t {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct tensor_args_t {
    Tensor input;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::nlp_concat_heads
