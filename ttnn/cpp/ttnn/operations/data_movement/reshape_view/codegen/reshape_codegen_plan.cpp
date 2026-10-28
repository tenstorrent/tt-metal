// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

// Largest slab an output stick is cut into.
constexpr uint32_t kMaxSlabBytes = 4096;
// Most slabs gathered behind one read barrier.
constexpr uint32_t kMaxSlabBatch = 8;
// A CB's received and acked page counters are 16 bits, so program creation rejects a CB of more pages.
constexpr uint64_t kMaxCbPages = std::numeric_limits<uint16_t>::max();

// Program creation starts every CB at a DRAM-aligned address, so a CB takes its size rounded up to that alignment.
uint64_t cb_footprint(uint64_t bytes) { return tt::round_up(bytes, tt::tt_metal::hal::get_dram_alignment()); }

// Largest power of two no greater than `n` (n > 0). Reduced CB depths step along powers of two so that small moves of
// the L1 frontier keep the same program and its cache key.
uint64_t power_of_two_floor(uint64_t n) { return std::bit_floor(n); }

uint64_t volume_before_last(const ttnn::Shape& shape) {
    uint64_t volume = 1;
    for (size_t i = 0; i + 1 < shape.rank(); ++i) {
        volume *= shape[i];
    }
    return volume;
}

uint32_t tile_bytes_of(const Tensor& input) {
    return input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
}

std::optional<Shape3> normalize(const ttnn::Shape& shape) {
    if (shape.rank() == 0) {
        return std::nullopt;
    }
    for (size_t i = 0; i < shape.rank(); ++i) {
        if (shape[i] == 0) {
            return std::nullopt;
        }
    }
    Shape3 out;
    out.w = shape[-1];
    if (shape.rank() >= 2) {
        out.h = shape[-2];
    }
    for (size_t i = 0; i + 2 < shape.rank(); ++i) {
        out.c *= shape[i];
    }
    return out;
}

std::optional<TileRowsChoice> fit_tile_rows(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned, uint64_t free_l1) {
    const uint32_t tile_bytes = tile_bytes_of(input);
    const auto fits = [&](const TileRowsChoice& choice) { return tile_rows_l1_bytes(choice, tile_bytes) <= free_l1; };
    if (fits(planned)) {
        return planned;
    }
    return plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits);
}

}  // namespace

std::optional<std::pair<Shape3, Shape3>> normalize_pair(const ttnn::Shape& in_shape, const ttnn::Shape& out_shape) {
    const auto in = normalize(in_shape);
    const auto out = normalize(out_shape);
    if (!in || !out || in->c * in->h * in->w != out->c * out->h * out->w) {
        return std::nullopt;
    }
    return std::make_pair(*in, *out);
}

tt::tt_metal::TensorSpec output_spec(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const tt::tt_metal::MemoryConfig& output_mem_config) {
    return tt::tt_metal::TensorSpec(
        logical_output_shape,
        tt::tt_metal::TensorLayout(
            input.dtype(), tt::tt_metal::PageConfig(input.layout(), input.tensor_spec().tile()), output_mem_config));
}

uint32_t output_aligned_page(const Tensor& input, const tt::tt_metal::TensorSpec& spec) {
    const uint32_t alignment = input.device()->allocator()->get_alignment(spec.memory_config().buffer_type());
    return static_cast<uint32_t>(tt::align(spec.compute_page_size_bytes(), alignment));
}

WorkCounts count_units(uint32_t grid_cores, uint32_t units) {
    if (units == 0 || grid_cores == 0) {
        return WorkCounts{};
    }
    const uint32_t num_cores = std::min(units, grid_cores);
    const uint32_t per_core = units / num_cores;
    if (units % num_cores == 0) {
        return WorkCounts{.units = units, .num_cores = num_cores, .units_group1 = per_core};
    }
    return WorkCounts{.units = units, .num_cores = num_cores, .units_group1 = per_core + 1, .units_group2 = per_core};
}

WorkSplit split_units(const tt::tt_metal::CoreRangeSet& grid, uint32_t units) {
    auto [num_cores, all_cores, group1, group2, units1, units2] = tt::tt_metal::split_work_to_cores(grid, units, false);
    const WorkCounts counts = count_units(static_cast<uint32_t>(grid.num_cores()), units);
    TT_FATAL(
        num_cores == counts.num_cores && units1 == counts.units_group1 && units2 == counts.units_group2,
        "Reshape work split over {} cores differs from the planned {} cores",
        num_cores,
        counts.num_cores);
    return WorkSplit{num_cores, all_cores, group1, group2, units1, units2};
}

tt::tt_metal::CoreRangeSet reshape_core_grid(
    const Tensor& input, const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids) {
    if (sub_core_grids.has_value()) {
        return *sub_core_grids;
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    return tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
}

uint64_t static_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    auto* device = input.device();
    // The same ceiling the live side caps at (available_program_l1_capacity): the bank already excludes the allocator
    // base and any reserved L1_SMALL region.
    uint64_t budget = device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1);

    const auto l1_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::L1);
    const auto l1_alignment = device->allocator()->get_alignment(tt::tt_metal::BufferType::L1);
    const auto subtract_footprint = [&](const tt::tt_metal::TensorSpec& spec) {
        const uint64_t footprint = spec.compute_consumed_memory_bytes_per_bank(l1_alignment, l1_banks);
        budget = budget > footprint ? budget - footprint : 0;
    };
    if (input.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
        subtract_footprint(input.tensor_spec());
    }
    if (output_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::L1) {
        subtract_footprint(output_spec);
    }
    return budget;
}

std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, const TileRowsChoice& planned) {
    return fit_tile_rows(input, logical_output_shape, planned, get_max_l1_space(input));
}

std::optional<TileMappedChoice> fit_tile_mapped(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, uint64_t l1_budget) {
    const uint32_t tile_bytes = tile_bytes_of(input);
    const auto layout = tile_mapped_layout(
        input.logical_shape(),
        logical_output_shape,
        static_cast<uint32_t>(input.element_size()),
        tile_bytes,
        grid_cores,
        l1_budget);
    if (!layout.has_value()) {
        return std::nullopt;
    }
    return TileMappedChoice{.layout = *layout, .input_depth = tile_mapped_input_depth(*layout, tile_bytes, l1_budget)};
}

std::optional<TileChoice> live_tile_choice(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t grid_cores, const LaunchPlan& planned) {
    const uint64_t free_l1 = get_max_l1_space(input);
    if (planned.tile_route == TileRoute::TileRows) {
        if (auto rows = fit_tile_rows(input, logical_output_shape, planned.tile_rows, free_l1)) {
            return TileChoice{std::move(*rows)};
        }
    }
    if (auto mapped = fit_tile_mapped(input, logical_output_shape, grid_cores, free_l1)) {
        return TileChoice{*mapped};
    }
    return std::nullopt;
}

std::optional<RmPlan> rm_geometry(
    const Tensor& input, const ttnn::Shape& logical_output_shape, uint32_t out_aligned_page, uint32_t grid_cores) {
    const auto& in_shape = input.logical_shape();
    if (in_shape.rank() == 0 || logical_output_shape.rank() == 0) {
        return std::nullopt;
    }
    const uint64_t num_old_sticks = volume_before_last(in_shape);
    const uint64_t num_new_sticks = volume_before_last(logical_output_shape);
    const uint32_t elem = input.element_size();

    RmPlan plan;
    plan.old_stick = in_shape[-1] * elem;
    plan.new_stick = logical_output_shape[-1] * elem;
    if (plan.old_stick == 0 || plan.new_stick == 0 || num_old_sticks == 0 || num_new_sticks == 0) {
        return std::nullopt;
    }
    // The work split and the kernels' page ids are 32-bit.
    constexpr uint64_t kMaxSticks = std::numeric_limits<uint32_t>::max();
    if (num_old_sticks > kMaxSticks || num_new_sticks > kMaxSticks) {
        return std::nullopt;
    }
    // A repartition that keeps the stick size is a metadata view and never reaches the kernels.
    if (plan.old_stick == plan.new_stick) {
        return std::nullopt;
    }
    const uint32_t big = std::max(plan.old_stick, plan.new_stick);
    const uint32_t small = std::min(plan.old_stick, plan.new_stick);
    if (big % small != 0) {
        return std::nullopt;
    }
    plan.ratio = big / small;

    plan.old_aligned = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    plan.new_aligned = out_aligned_page;
    plan.can_coalesce = plan.old_stick == plan.old_aligned && plan.new_stick == plan.new_aligned;
    plan.split_by_old = plan.old_stick > plan.new_stick;
    plan.split_nonaligned = plan.split_by_old && !plan.can_coalesce;

    const uint64_t split_sticks = plan.split_by_old ? num_old_sticks : num_new_sticks;
    plan.split = count_units(grid_cores, static_cast<uint32_t>(split_sticks));

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    plan.cb_page = plan.can_coalesce ? plan.new_stick : tt::round_up(plan.new_stick, l1_alignment);
    plan.scratch_bytes = plan.can_coalesce ? 0 : kRmScratchSticks * (plan.old_aligned + kRmScratchSlackBytes);
    return plan;
}

std::optional<RmPlan> fit_rm(RmPlan plan, uint64_t l1_budget) {
    const uint32_t max_sticks = std::max(plan.split.units_group1, plan.split.units_group2);
    // CB pages produced by one split stick; a push is always a whole number of these.
    const uint64_t pages_per_split_stick = plan.split_by_old ? plan.ratio : 1;
    const uint64_t full_depth = static_cast<uint64_t>(max_sticks) * pages_per_split_stick * (plan.can_coalesce ? 1 : 2);
    const uint64_t scratch_footprint = plan.scratch_bytes == 0 ? 0 : cb_footprint(plan.scratch_bytes);
    // Bytes the stick CB may take: live L1, and no more pages than the 16-bit CB page counters hold. Rounded down to
    // the CB alignment, a ring that fits it also fits once aligned.
    const uint64_t ring_l1 = l1_budget > scratch_footprint ? l1_budget - scratch_footprint : 0;
    const uint64_t ring_budget =
        tt::round_down(std::min(ring_l1, kMaxCbPages * plan.cb_page), tt::tt_metal::hal::get_dram_alignment());

    uint64_t depth = full_depth;
    plan.push_sticks = 0;
    if (full_depth * plan.cb_page > ring_budget) {
        // Two pushes in flight keep reader and writer overlapped; fewer is not worth running.
        const uint64_t push_bytes = 2 * pages_per_split_stick * plan.cb_page;
        if (plan.can_coalesce) {
            // Every core must push the same, evenly dividing stick count so the ring never wraps mid-push. The push
            // is the largest power of two dividing that count within both the read cap (one stick is always allowed)
            // and L1.
            const uint32_t common = std::gcd(plan.split.units_group1, plan.split.units_group2);
            const uint32_t split_stick_bytes = plan.split_by_old ? plan.old_stick : plan.new_stick;
            const uint64_t fit_limit = ring_budget / push_bytes;
            const uint64_t read_limit = std::max<uint64_t>(1, kMaxReadSize / split_stick_bytes);
            const uint64_t limit = std::min({uint64_t{1} << std::countr_zero(common), fit_limit, read_limit});
            if (limit == 0) {
                return std::nullopt;
            }
            plan.push_sticks = static_cast<uint32_t>(power_of_two_floor(limit));
            depth = 2 * static_cast<uint64_t>(plan.push_sticks) * pages_per_split_stick;
        } else {
            const uint64_t push_pairs = ring_budget / push_bytes;
            if (push_pairs == 0) {
                return std::nullopt;
            }
            depth = 2 * power_of_two_floor(push_pairs) * pages_per_split_stick;
        }
    }
    plan.cb_depth = static_cast<uint32_t>(depth);
    plan.l1_bytes = cb_footprint(depth * plan.cb_page) + scratch_footprint;
    return plan;
}

std::optional<RmArbitraryPlan> rm_arbitrary_geometry(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    uint32_t out_aligned_page,
    tt::tt_metal::BufferType output_buffer,
    uint32_t grid_cores) {
    const auto& in_shape = input.logical_shape();
    if (in_shape.rank() == 0 || logical_output_shape.rank() == 0) {
        return std::nullopt;
    }
    const uint64_t num_new_sticks = volume_before_last(logical_output_shape);
    const uint32_t elem = input.element_size();

    RmArbitraryPlan plan;
    plan.old_stick = in_shape[-1] * elem;
    plan.new_stick = logical_output_shape[-1] * elem;
    if (plan.old_stick == 0 || plan.new_stick == 0 || num_new_sticks == 0) {
        return std::nullopt;
    }
    // The kernels address the packed byte stream with 32-bit offsets.
    constexpr uint64_t kMaxOffset = std::numeric_limits<uint32_t>::max();
    if (num_new_sticks * plan.new_stick > kMaxOffset) {
        return std::nullopt;
    }

    plan.in_alignment = input.buffer()->alignment();
    const uint32_t out_alignment = input.device()->allocator()->get_alignment(output_buffer);
    plan.old_page = static_cast<uint32_t>(input.buffer()->aligned_page_size());
    plan.new_page = out_aligned_page;
    // A window read spans its bytes plus up to one alignment unit each side and must stay a single NOC transaction.
    const uint32_t burst = tt::tt_metal::hal::get_noc_max_burst_size_bytes();
    if (burst <= 2 * plan.in_alignment) {
        return std::nullopt;
    }
    plan.slab = tt::round_down(std::min(kMaxSlabBytes, burst - 2 * plan.in_alignment), out_alignment);
    if (plan.slab == 0) {
        return std::nullopt;
    }
    plan.slabs_per_stick = tt::div_up(plan.new_stick, plan.slab);
    const uint64_t units = num_new_sticks * plan.slabs_per_stick;
    if (units > kMaxOffset) {
        return std::nullopt;
    }

    const uint32_t slab_bytes = std::min(plan.slab, plan.new_stick);
    plan.slot = tt::round_up(slab_bytes, tt::tt_metal::hal::get_l1_alignment());
    // A slab spans at most slab_bytes / old_stick + 2 input pages, one window each.
    plan.region_stride =
        tt::round_up(slab_bytes + ((slab_bytes / plan.old_stick) + 2) * 2 * plan.in_alignment, plan.in_alignment);
    plan.split = count_units(grid_cores, static_cast<uint32_t>(units));
    return plan;
}

std::optional<RmArbitraryPlan> fit_rm_arbitrary(RmArbitraryPlan plan, uint64_t l1_budget) {
    plan.batch = 0;
    for (uint32_t batch = kMaxSlabBatch; batch >= 1; batch /= 2) {
        // One extra alignment unit lets the reader align the scratch base up.
        const uint32_t scratch = (batch * plan.region_stride) + plan.in_alignment;
        const uint64_t bytes = cb_footprint(scratch) + cb_footprint(2ull * batch * plan.slot);
        if (bytes <= l1_budget) {
            plan.batch = batch;
            plan.scratch_bytes = scratch;
            plan.l1_bytes = bytes;
            break;
        }
    }
    if (plan.batch == 0) {
        return std::nullopt;
    }
    return plan;
}

std::optional<RmChoice> live_rm_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    tt::tt_metal::BufferType output_buffer,
    uint32_t grid_cores,
    const LaunchPlan& planned) {
    const uint64_t free_l1 = get_max_l1_space(input);
    if (planned.rm.has_value()) {
        if (auto stick = fit_rm(*planned.rm, free_l1)) {
            return RmChoice{*stick};
        }
    }
    auto slab =
        planned.rm_arbitrary.has_value()
            ? planned.rm_arbitrary
            : rm_arbitrary_geometry(input, logical_output_shape, planned.out_aligned_page, output_buffer, grid_cores);
    if (slab.has_value()) {
        if (auto fitted = fit_rm_arbitrary(*slab, free_l1)) {
            return RmChoice{*fitted};
        }
    }
    return std::nullopt;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
