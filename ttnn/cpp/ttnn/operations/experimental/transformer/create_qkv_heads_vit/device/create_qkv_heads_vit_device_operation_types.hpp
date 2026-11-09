// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CreateQkvHeadsVitParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct CreateQkvHeadsVitInputs {
    Tensor input_tensor;
    std::optional<std::vector<std::optional<Tensor>>> optional_output_tensors;
};

using CreateQkvHeadsVitResult = std::vector<Tensor>;

using CreateQkvHeadsVitResultSpec = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::experimental::prim
