// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/host_buffer/functions.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/tile.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/core.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

Tensor to_row_major_layout(
    const Tensor& tensor_arg,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    float pad_value = 0.0f);

Tensor to_tile_layout(
    const Tensor& tensor_arg,
    const tt::tt_metal::Tile& tile = tt::tt_metal::Tile{},
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    float pad_value = 0.0f);

[[deprecated("Use to_row_major_layout() or to_tile_layout(tile)")]]
Tensor to_layout(
    const Tensor& tensor_arg,
    Layout layout,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt,
    float pad_value = 0.0f);

}  // namespace ttnn
