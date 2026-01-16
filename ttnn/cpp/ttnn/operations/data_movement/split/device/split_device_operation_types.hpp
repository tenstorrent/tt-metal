// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::split {

struct SplitParams {
    int num_splits{};
    int dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct SplitInputs {
    Tensor input;
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::data_movement::split
