// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_supported.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::operations::data_movement::tilize_codegen {

namespace {
// Mirrors spec.py's _L1_RESERVE = 128 * 1024 (builder.py:251 "reserve for code/stacks"), the ONLY
// reserve tilize's own CB planners (_tilize_row_cb_plan / _tilize_block_cb_plan /
// _tilize_2d_column_cb_plan, via _select_tilize_cb_depths -> scale_cb_depths_to_l1) subtract from
// the arch's raw per-core L1 (ArchConfig.l1_size_bytes queried straight off the device, not offset
// by the allocator's own reserved base the way the untilize port's usable_l1_bytes() is).
constexpr uint32_t kL1Reserve = 128 * 1024;
}  // namespace

uint32_t usable_l1_bytes(const tt::tt_metal::IDevice* device) {
    uint32_t l1_total = device->l1_size_per_core();
    return l1_total > kL1Reserve ? l1_total - kL1Reserve : 0;
}

ImplementationSelector parse_implementation(const std::string& implementation) {
    if (implementation == "auto") {
        return ImplementationSelector::Auto;
    }
    if (implementation == "native") {
        return ImplementationSelector::Native;
    }
    if (implementation == "codegen") {
        return ImplementationSelector::Codegen;
    }
    TT_THROW("unknown implementation selector: '{}'", implementation);
}

// Correctness scope of the ported builders (row / block / 2D-column, all interleaved): ROW_MAJOR
// input, non-sharded input AND requested output memory config, dtype in the nightly sweep's
// coverage (manifest coverage.dtypes: bfloat16, float32, uint32, int32, uint16), tile-aligned
// width (native's own validate already TT_FATALs width % tile_width != 0; kernels here further
// assume a full TILE_H=32 stick group per tile-row via reader_tilize_block/reader_stick_
// interleaved_unified's H_per_tile constant, so a non-tile-aligned H is fine -- padding is
// handled by rounding Ht up, matching native).
//
// Per-leaf L1 admission: each of the three dispatch legs the program factory can reach
// (row / block / 2D-column) has its own CB-depth planner in spec.py, all ultimately bounded by
// the same device L1 budget (usable_l1_bytes) via scale_cb_depths_to_l1's floor-to-min(1)
// clamp -- so, unlike the block/2D legs (which always shrink to at least 1-deep, single-tile
// CBs and thus always fit), only the row leg's minimum-viable single-buffered plan
// (cb_in_depth=1, cb_out_depth>=1 tile) can still overflow when a single tile-ROW's CB_IN+CB_OUT
// bytes alone exceed the budget. That is the one bound this predicate must check per the porting
// guide: the row leg's static per-tile-row floor is 1*ts (in) + max(1, required_out)*out_ts, and
// if that floor alone does not fit, no scaling can save it and the whole call must fall back to
// native.
bool supported_by_codegen(
    const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config, tt::tt_metal::DataType output_dtype) {
    using tt::tt_metal::DataType;
    using tt::tt_metal::Layout;

    if (input.layout() != Layout::ROW_MAJOR) {
        return false;
    }
    static constexpr DataType kSupportedDtypes[] = {
        DataType::BFLOAT16, DataType::FLOAT32, DataType::UINT32, DataType::INT32, DataType::UINT16};
    if (std::find(std::begin(kSupportedDtypes), std::end(kSupportedDtypes), input.dtype()) ==
        std::end(kSupportedDtypes)) {
        return false;
    }
    if (std::find(std::begin(kSupportedDtypes), std::end(kSupportedDtypes), output_dtype) ==
        std::end(kSupportedDtypes)) {
        return false;
    }
    if (input.is_sharded()) {
        return false;
    }
    if (output_mem_config.is_sharded()) {
        return false;
    }
    // Native's own validate_on_program_cache_miss requires width % tile_width == 0 unconditionally
    // (non-retile case); mirror that here so a rejected case is reported at scope-gate time rather
    // than at the native structural TT_FATAL one call later.
    const auto& logical = input.logical_shape();
    if (logical[-1] % tt::constants::TILE_WIDTH != 0) {
        return false;
    }

    auto* device = input.device();
    if (device == nullptr) {
        // No device to query L1 against yet (host tensor) -- validate_on_program_cache_miss's own
        // device/buffer TT_FATALs are the ones that must fire for this case, not this predicate;
        // answer conservatively false so a host-side call routes to native rather than crashing.
        return false;
    }

    // Only the row-path builder is transliterated in this port; the block (builder.py's
    // uses_block_path, Wt > _BLOCK_THRESHOLD=32 with total_Ht<num_avail_cores) and 2D-column
    // (uses_2d_column_path) dispatch legs are NOT implemented here, so any shape that would fire
    // either must fall back to native rather than reach a factory that cannot serve it.
    constexpr uint32_t kBlockThreshold = 32;
    const uint32_t wt = (logical[-1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    uint32_t nc = 1;
    for (int i = 0; i + 2 < static_cast<int>(logical.rank()); ++i) {
        nc *= logical[i];
    }
    const uint32_t ht = (logical[-2] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    const uint32_t total_ht = nc * ht;
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_avail_cores = grid.x * grid.y;

    // uses_block_path's own guard: Wt > 32 and grid-underutilized.
    if (wt > kBlockThreshold && total_ht < num_avail_cores) {
        return false;
    }
    // uses_2d_column_path's guard (only reachable when uses_block_path is false, per spec.py):
    // fires iff _choose_tilize_2d_ncol finds a real divisor split, i.e. total_Ht < num_avail_cores,
    // Wt >= 2, and some divisor d>=2 of Wt satisfies total_Ht*d <= num_avail_cores. Transcribed
    // directly (not example shapes) so shapes that are merely underutilized but have no such
    // divisor (e.g. Wt prime and too large, or grid budget too small for d=2) stay in-scope.
    if (total_ht < num_avail_cores && wt >= 2) {
        uint64_t max_ncol = std::min<uint64_t>(num_avail_cores / total_ht, wt);
        for (uint32_t d = 2; d <= max_ncol; ++d) {
            if (wt % d == 0) {
                return false;
            }
        }
    }

    const uint32_t ts = tt::tt_metal::tile_size(tt::tt_metal::datatype_to_dataformat_converter(input.dtype()));
    const uint32_t out_ts = tt::tt_metal::tile_size(tt::tt_metal::datatype_to_dataformat_converter(output_dtype));

    const uint32_t usable_l1 = usable_l1_bytes(device);

    // Mirrors _tilize_row_cb_plan's minimal_work / write_batch=1 floor: whatever the row leg's
    // chunking degenerates to, cb_in_depth is at least 1 tile and cb_out_depth is at least
    // _tilize_required_cb_out(write_batch=1, compute_chunk=1) == 1 tile (write_batch collapses to
    // 1 whenever total_Ht > num_cores or the shape is minimal-work, both of which are the only
    // ways the per-core tile count could grow large enough to threaten this floor). A single
    // tile of CB_IN plus a single tile of CB_OUT is therefore the true floor every leg's planner
    // can be scaled down to; reject only if even that floor overflows.
    const uint64_t row_floor_bytes = static_cast<uint64_t>(ts) + static_cast<uint64_t>(out_ts);
    if (row_floor_bytes > usable_l1) {
        return false;
    }

    // Block / 2D-column legs additionally require Wt (tile columns) to actually be splittable
    // across the device's grid; _compute_2d_split already degrades to a single core (block_Wt=Wt)
    // when nothing better fits, so their floor is the SAME single-tile-row CB pair as the row
    // leg's, bounded by cb_block_limit = usable_l1 / (2*ts) >= 1 as long as the row floor above
    // holds. No separate check is needed for those two legs.

    return true;
}

bool supported_execution_controls(
    bool use_multicore, bool use_low_perf, const std::optional<CoreRangeSet>& sub_core_grids) {
    // Every codegen builder places work over the full compute_with_storage_grid_size() and has no
    // single-core / sub-core-grid variant, so a caller that restricted placement or forced
    // single-core execution must go to native, which honours those controls exactly.
    return use_multicore && !use_low_perf && !sub_core_grids.has_value();
}

// Perf-demote ledger: shapes that supported_by_codegen() already accepts (correct under codegen)
// but that a device-measured (DEVICE KERNEL DURATION, not e2e_perf) comparison found do not beat
// native. Empty until a verify run's routing.demotion_candidates names a general condition to
// route away.
bool is_demoted(
    const Tensor& /*input*/,
    const tt::tt_metal::MemoryConfig& /*output_mem_config*/,
    tt::tt_metal::DataType /*output_dtype*/) {
    return false;
}

}  // namespace ttnn::operations::data_movement::tilize_codegen
