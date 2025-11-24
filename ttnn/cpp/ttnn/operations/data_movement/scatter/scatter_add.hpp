// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::data_movement {

struct ScatterAddOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const std::optional<MemoryConfig>& opt_out_memory_config,
        const std::optional<CoreRangeSet>& sub_core_grid);
};

}  // namespace operations::data_movement

constexpr auto scatter_add =
    ttnn::register_operation<"ttnn::scatter_add", ttnn::operations::data_movement::ScatterAddOperation>();

}  // namespace ttnn
