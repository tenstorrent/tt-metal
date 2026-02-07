// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <tuple>

namespace ttnn::experimental::prim {

inline constexpr uint32_t HIDDEN_SIZE = 5120;

struct RepeatMulParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const MathFidelity math_fidelity;

    static constexpr auto attribute_names = std::forward_as_tuple("memory_config", "dtype", "math_fidelity");
    auto attribute_values() const { return std::forward_as_tuple(memory_config, dtype, math_fidelity); }
};

struct RepeatMulInputs {
    const Tensor& a;
    const Tensor& b;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("a", "b", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(a, b, preallocated_output); }
};

}  // namespace ttnn::experimental::prim
