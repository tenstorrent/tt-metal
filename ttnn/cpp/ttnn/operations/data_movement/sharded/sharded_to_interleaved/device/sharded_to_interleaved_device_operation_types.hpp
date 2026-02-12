// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tuple>

namespace ttnn::prim {

struct ShardedToInterleavedParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};
    uint32_t num_slices = 1;
    uint32_t slice_index = 0;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_mem_config", "output_dtype", "num_slices", "slice_index");
    auto attribute_values() const {
        return std::forward_as_tuple(output_mem_config, output_dtype, num_slices, slice_index);
    }
};

struct ShardedToInterleavedInputs {
    Tensor input_tensor;
    std::optional<Tensor> preallocated_output;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "preallocated_output");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, preallocated_output); }
};

}  // namespace ttnn::prim
