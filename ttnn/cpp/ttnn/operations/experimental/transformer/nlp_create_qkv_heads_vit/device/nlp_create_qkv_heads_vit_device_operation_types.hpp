// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct NlpCreateQkvHeadsVitParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct NlpCreateQkvHeadsVitInputs {
    Tensor input_tensor;
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors;
};

using NlpCreateQkvHeadsVitResult = std::vector<Tensor>;

using NlpCreateQkvHeadsVitResultSpec = std::vector<TensorSpec>;

}  // namespace ttnn::experimental::prim
