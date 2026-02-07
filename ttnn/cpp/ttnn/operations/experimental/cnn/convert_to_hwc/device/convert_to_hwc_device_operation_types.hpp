// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct ConvertToHwcParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config", "dtype");
    auto attribute_values() const { return std::forward_as_tuple(memory_config, dtype); }
};

struct ConvertToHwcInputs {
    const Tensor& input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

}  // namespace ttnn::experimental::prim
