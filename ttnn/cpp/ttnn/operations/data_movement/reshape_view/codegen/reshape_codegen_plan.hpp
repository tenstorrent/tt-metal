// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// Largest single NOC read the coalesced RM reader issues.
constexpr uint32_t kMaxReadSize = 2048;

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

WorkSplit split_units(const tt::tt_metal::CoreRangeSet& grid, uint32_t units);

// What the entry planned for one call. A program-cache miss builds from it instead of planning again; live L1 then
// picks the plan's size (live_tile_rows_choice, live_rm_choice).
struct LaunchPlan {
    // TILE.
    TileRowsChoice tile_rows;
    // ROW_MAJOR: the geometries routing built, not yet fitted to any budget. The slab geometry is only built when the
    // stick program does not fit the static window, or has no geometry.
    uint32_t out_aligned_page = 0;
    std::optional<RmPlan> rm;
    std::optional<RmArbitraryPlan> rm_arbitrary;
};

// A row-major program fitted to a budget: the stick program, or the slab program.
using RmChoice = std::variant<RmPlan, RmArbitraryPlan>;

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
// contract as live_tile_rows_choice: only the program hash and the factory call it, after the output is allocated.
std::optional<RmChoice> live_rm_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    tt::tt_metal::BufferType output_buffer,
    uint32_t grid_cores,
    const LaunchPlan& planned);

}  // namespace ttnn::operations::data_movement::reshape_codegen
