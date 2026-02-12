// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include <tuple>

namespace ttnn::prim {

struct ShardedToInterleavedPartialParams {
    uint32_t num_slices{};
    uint32_t slice_index{};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{};

    static constexpr auto attribute_names =
        std::forward_as_tuple("num_slices", "slice_index", "output_mem_config", "output_dtype");
    auto attribute_values() const {
        return std::forward_as_tuple(num_slices, slice_index, output_mem_config, output_dtype);
    }
};

struct ShardedToInterleavedPartialInputs {
    Tensor input_tensor;
    Tensor cache_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "cache_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, cache_tensor); }
};

}  // namespace ttnn::prim
