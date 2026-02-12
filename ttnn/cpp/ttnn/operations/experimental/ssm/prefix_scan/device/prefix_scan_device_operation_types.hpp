// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/base_types.hpp>
#include <tuple>

namespace ttnn::experimental::prim {

struct PrefixScanParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const MathFidelity math_fidelity;
};

struct PrefixScanInputs {
    Tensor a;
    Tensor bx;
    Tensor h_prev;

    static constexpr auto attribute_names = std::forward_as_tuple("a", "bx", "h_prev");
    auto attribute_values() const { return std::forward_as_tuple(a, bx, h_prev); }
};

}  // namespace ttnn::experimental::prim
