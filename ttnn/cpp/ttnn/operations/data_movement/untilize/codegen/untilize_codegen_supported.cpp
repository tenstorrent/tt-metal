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
    return device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
}

// Correctness scope of the ported builders: TILE input, interleaved (non-sharded) input AND
// requested output memory config, dtype in the nightly sweep's coverage (bfloat16, bfloat8_b).
//
// Tile-alignment is dtype-conditional (manifest port_scope.tile_aligned: [bfloat8_b]), matching
// both the manifest's own cases ([1,1,33,64] bf16 is scope:in via build_untilize_with_unpadding;
// the same shape in bfloat8_b is scope:out) and the attached op-orchestration source (tt-dm-codegen
// ops/untilize/untilize.py): UntilizeCodegen.untilize's `H % TILE_H != 0 or W % TILE_W != 0` branch
// does NOT reject non-tile-aligned bfloat16 -- it strips padding via build_untilize_with_unpadding,
// which this port implements as build_with_unpadding() in the program factory. Only bfloat8_b
// hitting that same branch is rejected here: untilize.py's _TILE_ONLY_DTYPES bridge first casts
// bf8_b -> bf16 via a native CopyCodegen dtype-cast this port does not implement, so non-aligned
// bfloat8_b alone stays out of scope and routes to native.
bool supported_by_codegen(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;

    if (input.layout() != Layout::TILE) {
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

    // Mirrors build_column_parallel + plan_cb_depths: that builder's CB plan is sized by the
    // busiest core's tile count, not by Wt, and plan_cb_depths() TT_FATALs once even the
    // single-buffer plan (cb_in + cb_out, one slot per tile each) exceeds the L1 budget. Reject
    // here so "auto" falls back to native instead of aborting inside program creation.
    auto column_parallel_plan_fits = [&](uint32_t wt) {
        auto* device = input.device();
        auto grid = device->compute_with_storage_grid_size();
        auto split = tt::tt_metal::split_work_to_cores(grid, wt, /*row_wise=*/true);
        uint32_t max_tiles_per_core =
            std::max(std::get<4>(split), std::get<3>(split).empty() ? 0u : std::get<5>(split));
        return 2ull * max_tiles_per_core * kTileSize <= usable_l1_bytes(device);
    };

    const auto& logical = input.logical_shape();
    const bool tile_aligned =
        logical[-2] % tt::constants::TILE_HEIGHT == 0 && logical[-1] % tt::constants::TILE_WIDTH == 0;

    if (!tile_aligned) {
        // bf8_b (block-float) cannot produce non-tile-aligned RM output directly -- the reference
        // casts to bf16 first (see file comment); that cast step is out of this port's scope.
        if (input.dtype() != DataType::BFLOAT16) {
            return false;
        }
        // Mirrors untilize.py's _wide_untilize_chunk_width guard for the with-unpadding path:
        // unlike build_untilize_tile, that builder has no column-parallel fallback, so ANY row
        // wide enough to overflow the L1 budget -- regardless of total_tile_rows -- is routed to a
        // slice/untilize/concat cascade outside this port's scope. Uses ceil(W/TILE_W) since W is
        // not tile-aligned here.
        uint32_t wt_ceil = (logical[-1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
        return 2ull * wt_ceil * kTileSize <= kWideChunkThreshold;
    }

    // Tile-aligned path: mirrors ops/untilize/untilize.py's wide-tensor guard for
    // build_untilize_tile -- a multi-tile-row input wide enough that a single tile-row would
    // overflow the slice+concat chunking threshold (~800KB for two double-buffered CBs at
    // 2048B/tile) is routed to a slice -> untilize -> concat cascade over unrelated builder
    // entries, out of scope here. Computed from the PADDED shape (Wt/Ht are physical, tile-grid
    // quantities), matching how the program factory itself derives Wt/total_tile_rows.
    const auto& padded_shape = input.padded_shape();
    uint32_t rank = padded_shape.rank();
    uint32_t w = padded_shape[-1];
    uint32_t h = padded_shape[-2];
    uint32_t wt = w / tt::constants::TILE_WIDTH;
    uint32_t nc = 1;
    for (uint32_t i = 0; i + 2 < rank; ++i) {
        nc *= padded_shape[i];
    }
    uint32_t total_tile_rows = nc * (h / tt::constants::TILE_HEIGHT);
    if (total_tile_rows > 1 && 2ull * wt * kTileSize > kWideChunkThreshold) {
        return false;
    }
    // A single tile-row skips the chunk cascade (build_column_parallel splits Wt across the grid
    // in one dispatch instead), so it is bounded by that builder's own per-core plan, not by the
    // whole-row threshold above.
    if (total_tile_rows == 1 && wt > 1 && !column_parallel_plan_fits(wt)) {
        return false;
    }

    return true;
}

bool supported_execution_controls(bool use_multicore, const std::optional<CoreRangeSet>& sub_core_grids) {
    return use_multicore && !sub_core_grids.has_value();
}

// Perf-demote ledger: shapes that supported_by_codegen() already accepts (correct under codegen)
// but that a device-measured (DEVICE KERNEL DURATION, not e2e_perf) comparison found do not beat
// native. The previous entries here (nightly/broaden_suite's bfloat8_b shapes) were re-measured
// under DEVICE KERNEL DURATION and found 20-55% AHEAD of native on every one -- the prior list came
// from codegen_untilize.py's e2e_perf, which is dispatch-overhead-dominated for these
// single-digit-microsecond kernels and doesn't reflect actual device time. Re-populate from a
// device kernel-duration comparison, never from e2e_perf, if a future case regresses.
//
// [6,4,102,93] bf16 DRAM was demoted here as a phase-2b seed (pre-port native-vs-generic reading);
// phase-7 re-measured it on the ported kernel (native/ported=3.42x, generic/ported=0.997x) and it
// clears the gate the seed was demoted ahead of -- removed per the phase-7 handoff. Empty demote
// set: every in-scope case is expected to run on codegen under auto.
bool is_demoted(const Tensor& /*input*/, const tt::tt_metal::MemoryConfig& /*output_mem_config*/) { return false; }

}  // namespace ttnn::operations::data_movement::untilize_codegen
