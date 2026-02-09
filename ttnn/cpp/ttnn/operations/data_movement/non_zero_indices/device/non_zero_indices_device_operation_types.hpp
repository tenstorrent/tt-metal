// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::nonzero {

struct NonzeroParams {
    tt::tt_metal::MemoryConfig output_memory_config;
};

struct NonzeroInputs {
    Tensor input;
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;

using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;

}  // namespace ttnn::operations::data_movement::nonzero
