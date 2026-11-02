// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/tile.hpp>

#include "ttnn/types.hpp"

namespace ttnn::operations::data_movement::detail {

// Verification-only: `ttnn.tilize` picks native or codegen itself, so these pin the choice. Bound only
// under `ttnn._ttnn.operations.data_movement`; use `ttnn::tilize` everywhere else.

ttnn::Tensor tilize_force_native(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt,
    bool use_multicore = true,
    bool use_low_perf = false,
    tt::tt_metal::Tile tile = tt::tt_metal::Tile(),
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

// Throws outside the codegen scope rather than falling back, so a comparison never measures native twice.
ttnn::Tensor tilize_force_codegen(
    const ttnn::Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    std::optional<DataType> output_dtype = std::nullopt);

}  // namespace ttnn::operations::data_movement::detail
