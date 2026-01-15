// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::transpose {

enum class TransposeOpDim { WH, HC, CN, NH, NW, CW };

enum class TransposeOpParallelizationStrategy { MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN };

struct TransposeParams {
    TransposeOpDim dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<float> pad_value;
};

struct TransposeInputs {
    Tensor input;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::data_movement::transpose
