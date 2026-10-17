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

// L1 window independent of live occupancy: the L1 bank (allocator range less L1_SMALL) minus the per-bank footprint
// of the input and output when they live in L1. Routing and validation gate on the smallest plan fitting it; which
// plan the program is built with is live_tile_rows_choice's decision.
uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// Per-core L1 the output takes once allocated; 0 unless it lives in L1.
uint64_t pending_output_l1_bytes(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// The plan the program is built with: `planned` when its buffers fit the L1 free below the lowest live allocation
// less `reserved_l1_bytes`, else the first smaller plan that does (plan_tile_rows_fitting); nullopt when none does.
// The program hash and the program factory call it after the output is allocated, with nothing allocating L1 between
// them, so the plan the hash keys is the plan the factory builds. Routing calls it before that allocation with the
// output reserved, so a call it admits has a plan at program creation.
std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const TileRowsChoice& planned,
    uint64_t reserved_l1_bytes = 0);

}  // namespace ttnn::operations::data_movement::reshape_codegen
