// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct PaddedSliceParams {
    const ttnn::Shape padded_slice_start;
    const ttnn::Shape padded_slice_end;
    const ttnn::Shape step;
    const tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("padded_slice_start", "padded_slice_end", "step", "output_mem_config");
    auto attribute_values() const {
        return std::forward_as_tuple(padded_slice_start, padded_slice_end, step, output_mem_config);
    }
};

struct PaddedSliceInputs {
    const Tensor& input;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input, preallocated_output); }
};

}  // namespace ttnn::experimental::prim
