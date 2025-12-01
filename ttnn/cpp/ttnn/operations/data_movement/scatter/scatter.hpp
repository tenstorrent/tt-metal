// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "scatter_enums.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::data_movement {

struct ScatterOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<std::string>& opt_reduction_string,
        const std::optional<CoreRangeSet>& sub_core_grid);
};

struct ScatterAddOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& output_memory_config,
        const std::optional<CoreRangeSet>& sub_core_grid);
};

}  // namespace operations::data_movement

constexpr auto scatter = ttnn::register_operation<"ttnn::scatter", ttnn::operations::data_movement::ScatterOperation>();

constexpr auto scatter_add =
    ttnn::register_operation<"ttnn::scatter_add", ttnn::operations::data_movement::ScatterAddOperation>();

}  // namespace ttnn
