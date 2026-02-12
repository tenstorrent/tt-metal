// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../scatter_enums.hpp"

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <tuple>

namespace ttnn::prim {

struct ScatterParams {
    // scatter dim
    const int32_t dim;
    const tt::tt_metal::MemoryConfig output_memory_config;
    // reduction applied to source values coming from repeating indices
    const ttnn::operations::data_movement::scatter::ScatterReductionType opt_reduction;
    const std::optional<CoreRangeSet> sub_core_grid;

    static constexpr auto attribute_names =
        std::forward_as_tuple("dim", "output_memory_config", "opt_reduction", "sub_core_grid");
    auto attribute_values() const {
        return std::forward_as_tuple(dim, output_memory_config, opt_reduction, sub_core_grid);
    }
};

struct ScatterInputs {
    const Tensor& input_tensor;
    const Tensor& index_tensor;
    const Tensor& src_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input_tensor", "index_tensor", "src_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input_tensor, index_tensor, src_tensor); }
};

}  // namespace ttnn::prim
