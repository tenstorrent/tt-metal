// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <variant>

#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor fold(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    bool use_transpose_as_fold = false,
    const std::optional<const ttnn::Shape>& output_shape = std::nullopt,
    std::variant<std::array<uint32_t, 2>, std::array<uint32_t, 4>, std::array<uint32_t, 6>> padding =
        std::array<uint32_t, 2>{0, 0},
    const std::optional<CoreRangeSet>& core_grid = std::nullopt,
    const std::optional<MemoryConfig>& override_memory_config = std::nullopt);

}  // namespace ttnn
