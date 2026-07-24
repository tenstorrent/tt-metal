// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_back {

struct PerTokenCastBackParams {
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::MemoryConfig output_memory_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_dtype", "output_memory_config");
    auto attribute_values() const { return std::forward_as_tuple(output_dtype, output_memory_config); }
};

struct PerTokenCastBackInputs {
    const Tensor& input_e4m3;
    const Tensor& input_scale;
};

}  // namespace ttnn::experimental::prim::per_token_cast_back
