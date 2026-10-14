// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// What the entry planned for one call; a program-cache miss builds from it instead of planning again.
struct LaunchPlan {
    TileRowsChoice tile_rows;
};

// Cores the op may use: `sub_core_grids` if given, else the device's compute grid.
tt::tt_metal::CoreRangeSet reshape_core_grid(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// The spec the op allocates its output from.
tt::tt_metal::TensorSpec output_spec(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config);

// L1 window independent of live occupancy: the allocator's L1 range minus the per-bank footprint of the input and
// output when they live in L1.
uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

}  // namespace ttnn::operations::data_movement::reshape_codegen
