// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_mapped_plan.hpp"
#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_tile_rows_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

// The gate admits only square 32x32 tiles of 16x16 faces. 64-bit so plan arithmetic on large shapes does not wrap.
inline constexpr uint64_t kTileDim = tt::constants::TILE_WIDTH;
inline constexpr uint64_t kFaceDim = tt::constants::FACE_WIDTH;
static_assert(tt::constants::TILE_HEIGHT == kTileDim && tt::constants::FACE_HEIGHT == kFaceDim);

// Planning walks every output tile row on the host; above this many tiles it costs seconds and hundreds of MB.
inline constexpr uint64_t kMaxPlannedOutTiles = 128 * 1024;

// Process-wide memo of planner results, which depend only on shapes and device constants. Evicts the least recently
// used entry beyond `capacity`.
template <typename Key, typename Value>
class PlanMemo {
public:
    explicit PlanMemo(size_t capacity) : capacity_(capacity) {}

    std::optional<Value> find(const Key& key) {
        std::lock_guard lock(mutex_);
        const auto it = index_.find(key);
        if (it == index_.end()) {
            return std::nullopt;
        }
        recency_.splice(recency_.begin(), recency_, it->second);
        return it->second->second;
    }

    void insert(const Key& key, Value value) {
        std::lock_guard lock(mutex_);
        if (index_.contains(key)) {
            return;
        }
        recency_.emplace_front(key, std::move(value));
        index_.emplace(key, recency_.begin());
        if (recency_.size() > capacity_) {
            index_.erase(recency_.back().first);
            recency_.pop_back();
        }
    }

private:
    using Entry = std::pair<Key, Value>;
    std::mutex mutex_;
    size_t capacity_;
    std::list<Entry> recency_;
    std::map<Key, typename std::list<Entry>::iterator> index_;
};

// Physical (C, H, W) model of a TILE shape: leading dims fold into C.
struct Shape3 {
    uint64_t c = 1;
    uint64_t h = 1;
    uint64_t w = 1;
};

inline uint64_t tile_pages(const Shape3& s) { return s.c * tt::div_up(s.h, kTileDim) * tt::div_up(s.w, kTileDim); }

// Both shapes as Shape3; nullopt when either has rank 0 or a zero dim, or the volumes differ.
std::optional<std::pair<Shape3, Shape3>> normalize_pair(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape);

// Largest single NOC read the coalesced RM reader issues.
constexpr uint32_t kMaxReadSize = 2048;

// The non-aligned stick reader gathers this many source sticks per NOC barrier into one scratch slot each. A slot is
// the aligned source page plus one NOC read-alignment unit of slack, so the read can keep the source's residue.
constexpr uint32_t kRmScratchSticks = 8;
// The reader's `read_alignment`: the largest NOC read alignment of any arch (64 B on BH, 32 B on WH), so its residue
// mask holds on both.
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
// of the input and output when they live in L1. Routing gates on the smallest plan fitting it, so the route never
// moves with live occupancy; which plan the program is built with is live_tile_choice's or live_rm_choice's decision.
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
