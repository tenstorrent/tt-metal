// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/codegen/reshape_codegen_plan.hpp"

#include <algorithm>
#include <limits>
#include <numeric>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

// Per-core RM scratch holds this many sticks, each with this much slack beyond its aligned page.
constexpr uint32_t kScratchSticks = 8;
constexpr uint32_t kScratchSlackBytes = 64;
// Largest slab an output stick is cut into.
constexpr uint32_t kMaxSlabBytes = 4096;
// Most slabs gathered behind one read barrier.
constexpr uint32_t kMaxSlabBatch = 8;

uint64_t volume_before_last(const ttnn::Shape& shape) {
    uint64_t volume = 1;
    for (size_t i = 0; i + 1 < shape.rank(); ++i) {
        volume *= shape[i];
    }
    return volume;
}

}  // namespace

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

uint64_t pending_output_l1_bytes(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec) {
    // require_constructible: output_spec is the spec the op allocates, so a failure to build it is a bug, not a
    // reason to reserve nothing and route a call the factory then rejects.
    return get_pending_l1_output_reservation(
        input,
        output_spec.padded_shape(),
        output_spec.memory_config(),
        output_spec.data_type(),
        output_spec.layout(),
        /*require_constructible=*/true);
}

std::optional<TileRowsChoice> live_tile_rows_choice(
    const Tensor& input,
    const ttnn::Shape& logical_output_shape,
    const TileRowsChoice& planned,
    uint64_t reserved_l1_bytes) {
    const uint32_t tile_bytes =
        input.tensor_spec().tile().get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const uint64_t max_l1 = get_max_l1_space(input);
    const uint64_t free_l1 = max_l1 > reserved_l1_bytes ? max_l1 - reserved_l1_bytes : 0;
    const auto fits = [&](const TileRowsChoice& choice) { return tile_rows_l1_bytes(choice, tile_bytes) <= free_l1; };
    if (fits(planned)) {
        return planned;
    }
    return plan_tile_rows_fitting(input.logical_shape(), logical_output_shape, fits);
}

uint64_t live_l1_budget(const Tensor& input) { return get_max_l1_space(input); }

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
    plan.scratch_bytes = plan.can_coalesce ? 0 : kScratchSticks * (plan.old_aligned + kScratchSlackBytes);
    return plan;
}

std::optional<RmPlan> fit_rm(RmPlan plan, uint64_t l1_budget) {
    const uint32_t max_sticks = std::max(plan.split.units_group1, plan.split.units_group2);
    // CB pages produced by one split stick; a push is always a whole number of these.
    const uint64_t pages_per_split_stick = plan.split_by_old ? plan.ratio : 1;
    const uint64_t full_depth = static_cast<uint64_t>(max_sticks) * pages_per_split_stick * (plan.can_coalesce ? 1 : 2);
    const auto fits = [&](uint64_t depth) { return depth * plan.cb_page + plan.scratch_bytes <= l1_budget; };

    uint64_t depth = full_depth;
    plan.push_sticks = 0;
    if (!fits(full_depth)) {
        // Two pushes in flight keep reader and writer overlapped; fewer is not worth running.
        if (plan.can_coalesce) {
            // Every core must push the same, evenly dividing stick count so the ring never wraps mid-push. The push
            // is the largest divisor of that count within both the read cap (one stick is always allowed) and L1.
            const uint32_t common = std::gcd(plan.split.units_group1, plan.split.units_group2);
            const uint32_t split_stick_bytes = plan.split_by_old ? plan.old_stick : plan.new_stick;
            const uint64_t push_bytes = 2 * pages_per_split_stick * plan.cb_page;
            const uint64_t fit_limit =
                l1_budget > plan.scratch_bytes ? (l1_budget - plan.scratch_bytes) / push_bytes : 0;
            const uint64_t read_limit = std::max<uint64_t>(1, kMaxReadSize / split_stick_bytes);
            uint64_t push = std::min({static_cast<uint64_t>(common), fit_limit, read_limit});
            while (push > 0 && common % push != 0) {
                --push;
            }
            if (push == 0) {
                return std::nullopt;
            }
            plan.push_sticks = static_cast<uint32_t>(push);
            depth = 2 * push * pages_per_split_stick;
        } else {
            const uint64_t slots = l1_budget > plan.scratch_bytes ? (l1_budget - plan.scratch_bytes) / plan.cb_page : 0;
            const uint64_t pushes = slots / pages_per_split_stick;
            if (pushes < 2) {
                return std::nullopt;
            }
            depth = pushes * pages_per_split_stick;
        }
    }
    plan.cb_depth = static_cast<uint32_t>(depth);
    plan.l1_bytes = depth * plan.cb_page + plan.scratch_bytes;
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
        const uint64_t bytes = scratch + (2ull * batch * plan.slot);
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

}  // namespace ttnn::operations::data_movement::reshape_codegen
