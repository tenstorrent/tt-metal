// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_mapped_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

inline constexpr uint64_t kTileDim = 32;

// Physical (C, H, W) model of a TILE shape: leading dims fold into C.
struct Shape3 {
    uint64_t c = 1;
    uint64_t h = 1;
    uint64_t w = 1;
};

inline uint64_t tile_pages(const Shape3& s) { return s.c * tt::div_up(s.h, kTileDim) * tt::div_up(s.w, kTileDim); }

// Both shapes as Shape3; nullopt when either has rank 0 or a zero dim, or the volumes differ.
std::optional<std::pair<Shape3, Shape3>> normalize_pair(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// A segment map fitted to a budget: the layout and its input CB depth.
struct TileMappedChoice {
    TileMappedLayout layout;
    uint32_t input_depth = 0;
};

// The TILE program a call runs. Tile rows needs both widths to be multiples of 32; the segment map takes any widths.
enum class TileRoute : uint32_t { TileRows = 0, TileMapped = 1 };

// What the entry planned for one call. A program-cache miss builds from it instead of planning again; live L1 then
// picks the plan's size (live_tile_choice).
struct LaunchPlan {
    // TILE: the route, with the full tile-rows plan when it takes tile rows. The segment map is planned from the
    // shapes alone.
    TileRoute tile_route = TileRoute::TileRows;
    TileRowsChoice tile_rows;
};

// A TILE program fitted to a budget, one alternative per TileRoute.
using TileChoice = std::variant<TileRowsChoice, TileMappedChoice>;

// Cores the op may use: `sub_core_grids` if given, else the device's compute grid.
tt::tt_metal::CoreRangeSet reshape_core_grid(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// The spec the op allocates its output from.
tt::tt_metal::TensorSpec output_spec(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config);

// L1 window independent of live occupancy: the L1 bank (allocator range less L1_SMALL) minus the per-bank footprint
// of the input and output when they live in L1. Routing gates on the smallest plan fitting it, so the route never
// moves with live occupancy; which plan the program is built with is live_tile_choice's decision.
uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// The plan the program is built with: `planned` when its buffers fit the L1 free below the lowest live allocation,
// else the first smaller plan that does (plan_tile_rows_fitting); nullopt when none does. The program hash and the
// program factory call it after the output is allocated, with nothing allocating L1 between them, so the plan the
// hash keys is the plan the factory builds.
std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned);

// The program of `planned.tile_route` sized to live L1: the largest tile-rows plan that fits, else the segment map
// with the deepest input CB that fits; its smallest plan needs no more L1 than native's, so routing need not look at
// live L1. nullopt when nothing fits. Same contract as live_tile_rows_choice.
std::optional<TileChoice> live_tile_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, const LaunchPlan& planned);

// The segment map fitted to `l1_budget`, nullopt when it exceeds its size caps or does not fit with one input slot.
std::optional<TileMappedChoice> fit_tile_mapped(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, uint64_t l1_budget);

}  // namespace ttnn::operations::data_movement::reshape_codegen
