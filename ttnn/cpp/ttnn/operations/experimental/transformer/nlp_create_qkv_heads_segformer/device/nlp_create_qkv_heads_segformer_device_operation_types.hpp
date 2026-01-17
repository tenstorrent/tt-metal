// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsSegformerParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpCreateQkvHeadsSegformerInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

using NlpCreateQkvHeadsSegformerResult = std::tuple<Tensor, Tensor, Tensor>;
using NlpCreateQkvHeadsSegformerResultSpec = std::tuple<TensorSpec, TensorSpec, TensorSpec>;

}  // namespace ttnn::experimental::prim
