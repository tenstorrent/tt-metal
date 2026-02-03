// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

enum class TransposeOpDim { WH, HC, CN, NH, NW, CW };

enum class TransposeOpParallelizationStrategy { MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN };

struct TransposeParams {
    TransposeOpDim dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
    float pad_value = 0.0f;
};

struct TransposeInputs {
    Tensor input;
};

}  // namespace ttnn::prim
