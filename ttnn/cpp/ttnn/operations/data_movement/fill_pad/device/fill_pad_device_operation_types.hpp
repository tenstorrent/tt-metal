// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct FillPadParams {
    float fill_value;
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("fill_value", "output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(fill_value, output_mem_config); }
};

struct FillPadInputs {
    Tensor input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

}  // namespace ttnn::prim
