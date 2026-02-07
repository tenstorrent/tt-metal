// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct NlpKvCacheLoadSliceParams {
    ttnn::Shape output_tensor_start;
    ttnn::Shape output_tensor_end;

    static constexpr auto attribute_names = std::forward_as_tuple("output_tensor_start", "output_tensor_end");
    auto attribute_values() const { return std::forward_as_tuple(output_tensor_start, output_tensor_end); }
};

struct NlpKvCacheLoadSliceInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::experimental::prim
