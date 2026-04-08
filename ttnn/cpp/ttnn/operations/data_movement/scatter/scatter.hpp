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

Tensor scatter(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<std::string>& opt_reduction_string = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

Tensor scatter_add(
    const Tensor& input_tensor,
    const int32_t& dim,
    const Tensor& index_tensor,
    const Tensor& source_tensor,
    const std::optional<MemoryConfig>& output_memory_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn
