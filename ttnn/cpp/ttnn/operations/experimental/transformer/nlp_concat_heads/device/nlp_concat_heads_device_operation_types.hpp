// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"

namespace ttnn::operations::experimental::nlp_concat_heads {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpConcatHeadsInputs {
    Tensor input;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::nlp_concat_heads
