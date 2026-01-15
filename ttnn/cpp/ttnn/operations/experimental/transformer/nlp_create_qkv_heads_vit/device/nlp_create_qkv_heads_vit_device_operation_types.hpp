// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_vit {

struct NlpCreateQkvHeadsVitParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpCreateQkvHeadsVitInputs {
    Tensor input_tensor;
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors;
};

using tensor_return_value_t = std::vector<Tensor>;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::transformer::nlp_create_qkv_heads_vit
