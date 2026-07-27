// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {

ttnn::Tensor permute(
    const ttnn::Tensor& input_tensor,
    const SmallVector<int64_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    float pad_value = 0.0f,
    // Confine the op to a subset of Tensix cores. Required under an active SubDevice:
    // program factories otherwise claim the full compute grid.
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

ttnn::Tensor permute(const ttnn::Tensor& input_tensor, const SmallVector<int64_t>& dims, float pad_value = 0.0f);

}  // namespace ttnn
