// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {
struct GatherParams {
    const int8_t dim;
    const bool sparse_grad;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const std::optional<CoreRangeSet> sub_core_grids;

    static constexpr auto attribute_names =
        std::forward_as_tuple("dim", "sparse_grad", "output_mem_config", "sub_core_grids");
    auto attribute_values() const { return std::forward_as_tuple(dim, sparse_grad, output_mem_config, sub_core_grids); }
};

struct GatherInputs {
    const Tensor& input_tensor;
    const Tensor& input_index_tensor;
    std::optional<Tensor> output_tensor;

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensor", "input_index_tensor", "output_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, input_index_tensor, output_tensor); }
};

}  // namespace ttnn::prim
