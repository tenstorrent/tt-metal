// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_mapped_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Largest single NOC read the coalesced RM reader issues.
constexpr uint32_t kMaxReadSize = 2048;

// The non-aligned stick reader gathers this many source sticks per NOC barrier into one scratch slot each. A slot is
// the aligned source page plus one NOC read-alignment unit of slack, so the read can keep the source's residue.
constexpr uint32_t kRmScratchSticks = 8;
constexpr uint32_t kRmScratchSlackBytes = 64;

// How `units` of work spread over a grid of `grid_cores` cores, as split_work_to_cores spreads them: `num_cores` cores
// in use, the first ones take `units_group1` units, the rest `units_group2`. Plans carry only these counts; the core
// sets are built when a program is.
struct WorkCounts {
    uint32_t units = 0;
    uint32_t num_cores = 0;
    uint32_t units_group1 = 0;
    uint32_t units_group2 = 0;
};

WorkCounts count_units(uint32_t grid_cores, uint32_t units);

// Work split over a core grid: cores in `group1` take `units_group1` units, cores in `group2` take `units_group2`.
struct WorkSplit {
    uint32_t num_cores = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet group1;
    tt::tt_metal::CoreRangeSet group2;
    uint32_t units_group1 = 0;
    uint32_t units_group2 = 0;
};

// Row-major plan: both stick widths divide one another, sticks are re-cut between them.
struct RmPlan {
    uint32_t old_stick = 0;
    uint32_t new_stick = 0;
    uint32_t old_aligned = 0;
    uint32_t new_aligned = 0;
    uint32_t ratio = 0;
    bool can_coalesce = false;
    bool split_by_old = false;
    bool split_nonaligned = false;
    uint32_t cb_page = 0;
    uint32_t cb_depth = 0;
    // Split sticks per coalesced push on every core; 0 keeps the per-core read merge (CB holds a core's whole share).
    uint32_t push_sticks = 0;
    uint32_t scratch_bytes = 0;
    uint64_t l1_bytes = 0;
    WorkCounts split;
};

// Row-major plan for stick widths that do not divide one another, or that do when the stick plan does not fit. Each
// output stick is cut into slabs (`slot` bytes, one output CB page) read from every input page they span into a
// scratch region of `region_stride` bytes, then copied out logical bytes only. The output CB holds two batches of
// `batch` slabs.
struct RmArbitraryPlan {
    uint32_t old_stick = 0;
    uint32_t old_page = 0;
    uint32_t new_stick = 0;
    uint32_t new_page = 0;
    uint32_t in_alignment = 0;
    uint32_t slab = 0;
    uint32_t slabs_per_stick = 0;
    uint32_t slot = 0;
    uint32_t region_stride = 0;
    uint32_t batch = 0;
    uint32_t scratch_bytes = 0;
    uint64_t l1_bytes = 0;
    WorkCounts split;
};

// A segment map fitted to a budget: the layout and its input CB depth.
struct TileMappedChoice {
    TileMappedLayout layout;
    uint32_t input_depth = 0;
};

WorkSplit split_units(const tt::tt_metal::CoreRangeSet& grid, uint32_t units);

// The TILE program a call runs. Tile rows needs both widths to be multiples of 32; the segment map takes any widths.
enum class TileRoute : uint32_t { TileRows = 0, TileMapped = 1 };

// What the entry planned for one call. A program-cache miss builds from it instead of planning again; live L1 then
// picks the plan's size (live_tile_choice, live_rm_choice).
struct LaunchPlan {
    // TILE: the route, with the full tile-rows plan when it takes tile rows. The segment map is planned from the
    // shapes alone.
    TileRoute tile_route = TileRoute::TileRows;
    TileRowsChoice tile_rows;
    // ROW_MAJOR: the geometries routing built, not yet fitted to any budget. The slab geometry is only built when the
    // stick program does not fit the static window, or has no geometry.
    uint32_t out_aligned_page = 0;
    std::optional<RmPlan> rm;
    std::optional<RmArbitraryPlan> rm_arbitrary;
};

// A row-major program fitted to a budget: the stick program, or the slab program.
using RmChoice = std::variant<RmPlan, RmArbitraryPlan>;

// A TILE program fitted to a budget, one alternative per TileRoute.
using TileChoice = std::variant<TileRowsChoice, TileMappedChoice>;

// Cores the op may use: `sub_core_grids` if given, else the device's compute grid.
tt::tt_metal::CoreRangeSet reshape_core_grid(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids = std::nullopt);

// The spec the op allocates its output from.
tt::tt_metal::TensorSpec output_spec(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config);

// Page pitch the output buffer will have, for the gate and program hash where no buffer exists yet. The factory
// takes the value from the allocated buffer and checks it against this one.
uint32_t output_aligned_page(const Tensor& input, const tt::tt_metal::TensorSpec& spec);

// L1 window independent of live occupancy: the L1 bank (allocator range less L1_SMALL) minus the per-bank footprint
// of the input and output when they live in L1. Routing and validation gate on the smallest plan fitting it, so the
// route never moves with live occupancy; which plan the program is built with is live_tile_choice's or
// live_rm_choice's decision.
uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// The plan the program is built with: `planned` when its buffers fit the L1 free below the lowest live allocation,
// else the first smaller plan that does (plan_tile_rows_fitting); nullopt when none does. The program hash and the
// program factory call it after the output is allocated, with nothing allocating L1 between them, so the plan the
// hash keys is the plan the factory builds.
std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned);

// The program of `planned.tile_route` sized to the L1 free below the lowest live allocation: the largest tile-rows
// plan (live_tile_rows_choice), or the segment map, sliced where slicing pays off and fits, with the deepest input CB
// that fits. When no tile-rows plan fits, the segment map: its smallest plan, unsliced with one input slot, needs no
// more L1 than native's tiled reshape (tile_mapped_input_depth), so routing need not look at live L1. nullopt when
// nothing fits. Same contract as live_tile_rows_choice.
std::optional<TileChoice> live_tile_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, const LaunchPlan& planned);

// The segment map fitted to `l1_budget`, nullopt when it exceeds its size caps or does not fit with one input slot.
std::optional<TileMappedChoice> fit_tile_mapped(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, uint64_t l1_budget);

// Each row-major program is planned in two steps. The geometry step depends only on the shapes, dtype, memory
// configs and grid, and returns nullopt when the case is outside what the kernels handle; the fit step sizes the CB
// depth or batch to `l1_budget` and returns nullopt when the CBs exceed it at their minimum depth. The gate fits one
// geometry to static_l1_budget, live_rm_choice fits it to live L1.
std::optional<RmPlan> rm_geometry(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t out_aligned_page, uint32_t grid_cores);

std::optional<RmPlan> fit_rm(RmPlan plan, uint64_t l1_budget);

std::optional<RmArbitraryPlan> rm_arbitrary_geometry(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    uint32_t out_aligned_page,
    tt::tt_metal::BufferType output_buffer,
    uint32_t grid_cores);

std::optional<RmArbitraryPlan> fit_rm_arbitrary(RmArbitraryPlan plan, uint64_t l1_budget);

// The row-major program the factory builds: the stick program at the deepest CB that fits the L1 free below the
// lowest live allocation, else the slab program at the largest batch that does; nullopt when neither fits. Same
// contract as live_tile_rows_choice.
std::optional<RmChoice> live_rm_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    tt::tt_metal::BufferType output_buffer,
    uint32_t grid_cores,
    const LaunchPlan& planned);

}  // namespace ttnn::operations::data_movement::reshape_codegen
