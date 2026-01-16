// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::transformer::rotate_half {

struct RotateHalfParams {
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct RotateHalfInputs {
    Tensor input;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::transformer::rotate_half
