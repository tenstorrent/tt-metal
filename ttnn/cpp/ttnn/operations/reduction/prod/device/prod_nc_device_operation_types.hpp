// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct ProdNcParams {
    int64_t dim;

    static constexpr auto attribute_names = std::forward_as_tuple("dim");
    auto attribute_values() const { return std::forward_as_tuple(dim); }
};

struct ProdNcInputs {
    Tensor input;
    Tensor output;  // Note: output is passed as input (inplace pattern)

    static constexpr auto attribute_names = std::forward_as_tuple("input", "output");
    auto attribute_values() const { return std::forward_as_tuple(input, output); }
};

}  // namespace ttnn::prim
