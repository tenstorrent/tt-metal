// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_segformer {

struct NlpCreateQkvHeadsSegformerParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpCreateQkvHeadsSegformerInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor, Tensor>;
using spec_return_value_t = std::tuple<TensorSpec, TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_segformer
