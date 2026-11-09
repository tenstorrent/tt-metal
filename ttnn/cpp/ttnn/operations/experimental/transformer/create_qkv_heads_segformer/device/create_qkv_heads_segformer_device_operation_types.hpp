// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct CreateQkvHeadsSegformerParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct CreateQkvHeadsSegformerInputs {
    Tensor input_tensor;
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

using CreateQkvHeadsSegformerResult = std::tuple<Tensor, Tensor, Tensor>;
using CreateQkvHeadsSegformerResultSpec =
    std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

}  // namespace ttnn::experimental::prim
