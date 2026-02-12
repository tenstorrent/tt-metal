// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

enum class TransposeOpDim { WH, HC, CN, NH, NW, CW };

enum class TransposeOpParallelizationStrategy { MULTI_CORE_WH, MULTI_CORE_HC, MULTI_CORE_CN };

struct TransposeParams {
    TransposeOpDim dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
    float pad_value = 0.0f;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "output_mem_config", "pad_value");
    auto attribute_values() const { return std::forward_as_tuple(dim, output_mem_config, pad_value); }
};

struct TransposeInputs {
    Tensor input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

}  // namespace ttnn::prim
