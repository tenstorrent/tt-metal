// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <tuple>

namespace ttnn::experimental::prim {

struct HcSumReduceParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const MathFidelity math_fidelity;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config", "dtype", "math_fidelity");
    auto attribute_values() const { return std::forward_as_tuple(memory_config, dtype, math_fidelity); }
};

struct HcSumReduceInputs {
    Tensor input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

}  // namespace ttnn::experimental::prim
