// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshard {

struct ReshardParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ReshardInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::data_movement::reshard
