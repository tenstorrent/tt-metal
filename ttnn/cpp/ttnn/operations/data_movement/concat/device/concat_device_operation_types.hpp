// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include <tuple>

namespace ttnn::prim {

struct ConcatParams {
    uint32_t dim;
    unsigned int groups;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::CoreRangeSet> sub_core_grids;

    static constexpr auto attribute_names = std::forward_as_tuple("dim", "groups", "output_mem_config", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(dim, groups, output_mem_config, sub_core_grids); }
};

struct ConcatInputs {
    std::vector<Tensor> input_tensors;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensors");
    auto attribute_values() const { return std::forward_as_tuple(input_tensors); }
};

}  // namespace ttnn::prim
