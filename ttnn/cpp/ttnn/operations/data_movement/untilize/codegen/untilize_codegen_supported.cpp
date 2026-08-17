// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement::untilize_codegen {

uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device) {
    // Same accounting as ProgramImpl::validate_circular_buffer_region(): statically allocated CBs
    // grow upward from the allocator's base L1 address, L1 buffers are allocated downward from the
    // top of L1, and that validator TT_THROWs as soon as the CB region end passes
    // lowest_occupied_compute_l1_address(). Budget against the same frontier so a plan this gate
    // accepts is one the program validator will accept too. Mirrors get_max_l1_space() in
    // data_movement/common/common.cpp.
    //
    // std::nullopt means nothing is resident in L1 yet, in which case the whole region above the
    // base is available.
    const uint32_t base = device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const auto lowest_occupied = device->lowest_occupied_compute_l1_address();
    const uint32_t ceiling =
        lowest_occupied.has_value() ? static_cast<uint32_t>(*lowest_occupied) : device->l1_size_per_core();
    return ceiling > base ? ceiling - base : 0;
}

// Correctness scope of the codegen path: TILE input, interleaved (non-sharded) input AND
// requested output memory config, dtype bfloat16 or bfloat8_b.
//
// Tile-alignment is dtype-conditional. Non-tile-aligned bfloat16 is in scope -- build_with_unpadding()
// in the program factory strips the padding directly. Non-tile-aligned bfloat8_b is not: serving it
// would require first casting bf8_b -> bf16, a step this implementation does not have, so it routes
// to native.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;

    if (input.layout() != Layout::TILE) {
        return false;
    }
    // Every page-geometry term here and in the program factory is a hardcoded 32x32 quantity --
    // kTileSize below, and the wt / total_tile_rows divisions. An off-default tile shape changes
    // both the buffer's page size and its page count, so that plan mis-addresses it in both
    // directions. A transposed tile keeps both but permutes the datums the untilize kernels lift
    // out of each face, and nothing here configures the unpacker for that permutation. Declining
    // is not a correctness guarantee for either case: native reads the tile but does not serve an
    // off-default one reliably either. This only stops codegen claiming support it lacks.
    const auto tile = input.tensor_spec().tile();
    if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH ||
        tile.get_transpose_within_face() || tile.get_transpose_of_faces()) {
        return false;
    }
    if (input.dtype() != DataType::BFLOAT16 && input.dtype() != DataType::BFLOAT8_B) {
        return false;
    }
    if (input.is_sharded()) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }

    constexpr uint32_t kTileSize = 2048;  // bf16/bf8_b only in this scope
    constexpr uint64_t kWideChunkThreshold = 800'000;

    auto* device = input.device();

    // plan_cb_depths()'s floor tier: cb_in + cb_out at a single slot each. The factory has no
    // smaller plan to degrade to below this and TT_THROWs instead, so anything that does not clear
    // the floor has to be routed to native from here rather than aborting inside program creation.
    //
    // usable_l1_bytes() is budgeted against LIVE L1 occupancy, so this check is not a static
    // shape property: the same shape can clear it on an idle device and fail it alongside a
    // model's resident weight/trace buffers. That is deliberate -- it is the same budget
    // ProgramImpl::validate_circular_buffer_region() will hold the resulting program to.
    auto min_cb_plan_fits = [&](uint32_t pages_per_unit) {
        return 2ull * pages_per_unit * kTileSize <= usable_l1_bytes(device);
    };

    // Mirrors build_column_parallel + plan_cb_depths: that builder's CB plan is sized by the
    // busiest core's tile count, not by Wt.
    auto column_parallel_plan_fits = [&](uint32_t wt) {
        auto grid = device->compute_with_storage_grid_size();
        auto split = tt::tt_metal::split_work_to_cores(grid, wt, /*row_wise=*/true);
        uint32_t max_tiles_per_core =
            std::max(std::get<4>(split), std::get<3>(split).empty() ? 0u : std::get<5>(split));
        return min_cb_plan_fits(max_tiles_per_core);
    };

    // Wt / total_tile_rows are derived from the PADDED shape (Wt/Ht are physical, tile-grid
    // quantities), matching how the program factory itself derives them -- including for the
    // with-unpadding builder, which reads the full physical tile grid and only differs in its
    // writer.
    const auto& padded_shape = input.padded_shape();
    uint32_t rank = padded_shape.rank();
    uint32_t wt = padded_shape[-1] / tt::constants::TILE_WIDTH;
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= padded_shape[i];
    }
    uint32_t total_tile_rows = nc * (padded_shape[-2] / tt::constants::TILE_HEIGHT);

    const auto& logical = input.logical_shape();
    const bool tile_aligned =
        logical[-2] % tt::constants::TILE_HEIGHT == 0 && logical[-1] % tt::constants::TILE_WIDTH == 0;

    if (!tile_aligned) {
        // bf8_b (block-float) cannot produce non-tile-aligned RM output directly -- the reference
        // casts to bf16 first (see file comment); that cast step is out of this port's scope.
        if (input.dtype() != DataType::BFLOAT16) {
            return false;
        }
        // The with-unpadding path has no column-parallel fallback, so ANY row wide enough to
        // overflow the L1 budget -- regardless of total_tile_rows -- is out of scope and goes to
        // native. Uses ceil(W/TILE_W) since W is not tile-aligned here.
        uint32_t wt_ceil = (logical[-1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        if (2ull * wt_ceil * kTileSize > kWideChunkThreshold) {
            return false;
        }
        // build_with_unpadding plans its CBs on the physical Wt.
        return min_cb_plan_fits(wt);
    }

    // Tile-aligned path: a multi-tile-row input wide enough that a single tile-row would overflow
    // the chunking threshold (~800KB for two double-buffered CBs at 2048B/tile) needs a
    // slice -> untilize -> concat cascade this implementation does not have, so it is out of scope
    // here.
    if (total_tile_rows > 1 && 2ull * wt * kTileSize > kWideChunkThreshold) {
        return false;
    }
    // A single tile-row skips the chunk cascade (build_column_parallel splits Wt across the grid
    // in one dispatch instead), so it is bounded by that builder's own per-core plan, not by the
    // whole-row threshold above.
    if (total_tile_rows == 1 && wt > 1) {
        return column_parallel_plan_fits(wt);
    }

    // build_2d_column plans on wt/ncol and build_main_split on wt, so Wt bounds both. (ncol lives
    // in the factory's anonymous namespace; using the looser bound here can only over-route to
    // native, never under-route into a plan the factory cannot build.)
    return min_cb_plan_fits(wt);
}

bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

// Perf-demote ledger: shapes that supported_by_codegen() already accepts (correct under codegen)
// but that a device-measured (DEVICE KERNEL DURATION, not e2e_perf) comparison found do not beat
// native. The previous entries here (nightly/broaden_suite's bfloat8_b shapes) were re-measured
// under DEVICE KERNEL DURATION and found 20-55% AHEAD of native on every one -- the prior list came
// from an end-to-end timing, which is dispatch-overhead-dominated for these single-digit-microsecond
// kernels and doesn't reflect actual device time. Re-populate from a device kernel-duration
// comparison, never from an end-to-end one, if a future case regresses.
//
// [6,4,102,93] bf16 DRAM was demoted here on a pre-port reading; re-measured on the real kernel
// (native/ported=3.42x) it
// clears the gate the seed was demoted ahead of -- removed per the phase-7 handoff. Empty demote
// set: every in-scope case is expected to run on codegen under auto.
bool is_demoted(const Tensor& /*input*/, const tt::tt_metal::MemoryConfig& /*output_mem_config*/) { return false; }

}  // namespace ttnn::operations::data_movement::untilize_codegen
