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

// Mirrors builder.py's _compute_cb_block_limit: max tiles that fit in L1 for both CB0 + CB16
// (single-buffered).
uint32_t compute_cb_block_limit(uint32_t ts, uint32_t l1_size) { return l1_size / (2 * ts); }

// Mirrors builder.py's _compute_2d_split (lines 58-115): 2D block decomposition of (Ht x Wt) tile
// grid across cores. Returns (block_Wt, block_Ht, cores_w, cores_h); cliff sizes are not needed by
// the predicate (only uses_block_path's core-count / max(bHt,cHt) checks below consume this).
struct TwoDSplit {
    uint32_t block_wt;
    uint32_t block_ht;
    uint32_t cores_w;
    uint32_t cores_h;
    uint32_t cliff_wt;
    uint32_t cliff_ht;
};

TwoDSplit compute_2d_split(uint32_t ht, uint32_t wt, uint32_t num_avail_cores, uint32_t cb_block_limit) {
    const uint64_t total_tiles = static_cast<uint64_t>(ht) * wt;
    if (total_tiles <= cb_block_limit && num_avail_cores >= 1) {
        return {wt, ht, 1, 1, 0, 0};
    }

    const uint32_t max_block_wt = std::min(wt, cb_block_limit);

    bool found = false;
    uint32_t best_bw = 0, best_bh = 0, best_cw = 0, best_ch = 0;
    uint32_t best_ncores = 0;

    const uint32_t target_cw_limit = std::min(num_avail_cores, wt);
    for (uint32_t target_cw = 1; target_cw <= target_cw_limit; ++target_cw) {
        const uint32_t bw = (wt + target_cw - 1) / target_cw;
        if (bw > max_block_wt || bw == 0) {
            continue;
        }
        const uint32_t cw = (wt + bw - 1) / bw;
        if (cw == 0) {
            continue;
        }
        const uint32_t max_ch = num_avail_cores / cw;
        if (max_ch < 1) {
            continue;
        }
        const uint32_t ch_cand = std::min(ht, max_ch);
        const uint32_t bh = (ht + ch_cand - 1) / ch_cand;
        const uint32_t ch = (ht + bh - 1) / bh;

        const uint32_t nc = cw * ch;
        if (nc > num_avail_cores) {
            continue;
        }
        if (nc > best_ncores || (nc == best_ncores && found && (uint64_t)bw * bh < (uint64_t)best_bw * best_bh)) {
            best_bw = bw;
            best_bh = bh;
            best_cw = cw;
            best_ch = ch;
            best_ncores = nc;
            found = true;
        }
    }

    if (!found) {
        return {wt, ht, 1, 1, 0, 0};
    }
    const uint32_t cliff_wt = wt - (wt / best_bw) * best_bw;
    const uint32_t cliff_ht = ht - (ht / best_bh) * best_bh;
    return {best_bw, best_bh, best_cw, best_ch, cliff_wt, cliff_ht};
}

// Mirrors spec.py's uses_block_path (lines 452-473). Only reached after supported_by_codegen has
// already confirmed device/dtype/layout scope; l1_size is the device's raw usable L1 budget
// (matches spec.py's l1 default, which is the arch's per-core L1, not offset by kL1Reserve --
// _compute_cb_block_limit's own 2x headroom is the only slack the generator itself applies here).
bool uses_block_path(uint32_t wt, uint32_t total_ht, uint32_t num_avail_cores, uint32_t ts, uint32_t l1_size,
                      uint32_t block_threshold) {
    if (!(wt > block_threshold && total_ht < num_avail_cores)) {
        return false;
    }
    const uint32_t cb_block_limit = compute_cb_block_limit(ts, l1_size);
    const TwoDSplit split = compute_2d_split(total_ht, wt, num_avail_cores, cb_block_limit);
    if (std::max(split.block_ht, split.cliff_ht) > 1) {
        const bool row_needs_chunking = 2ull * wt * ts > l1_size;
        return row_needs_chunking && total_ht < num_avail_cores && (num_avail_cores / total_ht) >= 2;
    }
    return (static_cast<uint64_t>(split.cores_w) * split.cores_h) > total_ht;
}
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
    // uses_block_path, Wt > _BLOCK_THRESHOLD=32 with total_Ht<num_avail_cores, PLUS its
    // _compute_2d_split-gated secondary condition) and 2D-column (uses_2d_column_path) dispatch
    // legs are NOT implemented here, so any shape that would fire either must fall back to native
    // rather than reach a factory that cannot serve it.
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
    const uint32_t ts_for_split = tt::tt_metal::tile_size(input.dtype());
    const uint32_t l1_for_split = device->l1_size_per_core();

    // uses_block_path's own guard, including its secondary _compute_2d_split-derived condition
    // (spec.py lines 452-473) -- NOT just the initial Wt>32 gate, which alone over-rejects shapes
    // the real generator would fall through to uses_2d_column_path for.
    if (uses_block_path(wt, total_ht, num_avail_cores, ts_for_split, l1_for_split, kBlockThreshold)) {
        return false;
    }
    // uses_2d_column_path's guard (only reachable when uses_block_path is false, per spec.py):
    // fires iff _choose_tilize_2d_ncol finds a real divisor split, i.e. total_Ht < num_avail_cores,
    // Wt > 2 (uses_2d_column_path explicitly bails for Wt <= 2 -- narrow tensors don't carry enough
    // column work to amortize the 2D kernel geometry, so they stay on the row path), and some
    // divisor d>=2 of Wt satisfies total_Ht*d <= num_avail_cores. Transcribed directly (not example
    // shapes) so shapes that are merely underutilized but have no such divisor (e.g. Wt prime and
    // too large, or grid budget too small for d=2) stay in-scope.
    if (total_ht < num_avail_cores && wt > 2) {
        uint64_t max_ncol = std::min<uint64_t>(num_avail_cores / total_ht, wt);
        for (uint32_t d = 2; d <= max_ncol; ++d) {
            if (wt % d == 0) {
                return false;
            }
        }
    }

    const uint32_t ts = tt::tt_metal::tile_size(input.dtype());
    const uint32_t out_ts = tt::tt_metal::tile_size(output_dtype);

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
