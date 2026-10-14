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

// What the entry planned for one call: the full plan. A program-cache miss builds from it instead of planning again
// unless it no longer fits live L1 (live_tile_rows_choice).
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
// output when they live in L1. Routing and validation gate on the smallest plan fitting it; which plan the program
// is built with is live_tile_rows_choice's decision.
uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// The plan the program is built with: `planned` when its buffers fit the L1 free below the lowest live allocation,
// else the first smaller plan that does (plan_tile_rows_fitting); nullopt when none does. Reads live L1, so only the
// program hash and the program factory call it: both run after the output is allocated and nothing allocates L1
// between them, so the plan the hash keys is the plan the factory builds.
std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned);

}  // namespace ttnn::operations::data_movement::reshape_codegen
