// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/moe_compute/moe_core_placement.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <set>
#include <utility>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <umd/device/types/arch.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

// All worker-core placement helpers below are implementation details of
// select_moe_compute_cores() and are confined to this translation unit.
namespace {

using CoreCoordPairSet = std::set<std::pair<uint32_t, uint32_t>>;

constexpr uint32_t kMoEComputeMaxTilizeCores = 4;
// Legacy moe_compute combine pool was 2 columns wide (e.g. x=5,6 on WH).
constexpr uint32_t kMoEComputeCombineStripWidth = 2;

CoreCoordPairSet core_coords_to_pair_set(const std::vector<tt::tt_metal::CoreCoord>& cores) {
    CoreCoordPairSet result;
    for (const auto& core : cores) {
        result.insert({core.x, core.y});
    }
    return result;
}

std::vector<tt::tt_metal::CoreCoord> pick_worker_cores_row_major_avoiding(
    const CoreCoordPairSet& avoid, const tt::tt_metal::CoreCoord& worker_grid, uint32_t num_cores, uint32_t max_y_inclusive) {
    std::vector<tt::tt_metal::CoreCoord> picked;
    picked.reserve(num_cores);

    const uint32_t y_limit = std::min(max_y_inclusive + 1, static_cast<uint32_t>(worker_grid.y));
    for (uint32_t y = 0; y < y_limit && picked.size() < num_cores; ++y) {
        for (uint32_t x = 0; x < worker_grid.x && picked.size() < num_cores; ++x) {
            tt::tt_metal::CoreCoord candidate(x, y);
            if (!avoid.contains({candidate.x, candidate.y})) {
                picked.push_back(candidate);
            }
        }
    }

    return picked;
}

std::vector<tt::tt_metal::CoreCoord> pick_tilize_cores_in_upper_rows(
    const CoreCoordPairSet& avoid, const tt::tt_metal::CoreCoord& worker_grid, uint32_t num_cores, uint32_t min_y) {
    std::vector<tt::tt_metal::CoreCoord> picked;
    picked.reserve(num_cores);

    for (int y = static_cast<int>(worker_grid.y) - 1; y >= static_cast<int>(min_y) && picked.size() < num_cores; --y) {
        for (int x = static_cast<int>(worker_grid.x) - 1; x >= 0 && picked.size() < num_cores; --x) {
            tt::tt_metal::CoreCoord candidate(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
            if (!avoid.contains({candidate.x, candidate.y})) {
                picked.push_back(candidate);
            }
        }
    }

    return picked;
}

std::optional<CoreRange> find_combine_strip_avoiding(
    const CoreCoordPairSet& avoid, const tt::tt_metal::CoreCoord& worker_grid, uint32_t strip_height, uint32_t max_y_inclusive) {
    if (kMoEComputeCombineStripWidth > worker_grid.x || strip_height == 0) {
        return std::nullopt;
    }

    const uint32_t y_limit = std::min(max_y_inclusive + 1, static_cast<uint32_t>(worker_grid.y));

    // Prefer eastern columns (legacy pool was x=5,6 on WH).
    for (int sx = static_cast<int>(worker_grid.x) - static_cast<int>(kMoEComputeCombineStripWidth); sx >= 0; --sx) {
        for (uint32_t sy = 0; sy + strip_height <= y_limit; ++sy) {
            bool valid = true;
            for (uint32_t dy = 0; dy < strip_height && valid; ++dy) {
                for (uint32_t dx = 0; dx < kMoEComputeCombineStripWidth && valid; ++dx) {
                    if (avoid.contains({static_cast<uint32_t>(sx) + dx, sy + dy})) {
                        valid = false;
                    }
                }
            }
            if (valid) {
                return CoreRange(
                    {static_cast<uint32_t>(sx), sy},
                    {static_cast<uint32_t>(sx) + kMoEComputeCombineStripWidth - 1, sy + strip_height - 1});
            }
        }
    }
    return std::nullopt;
}

std::vector<tt::tt_metal::CoreCoord> pick_combine_cores_from_strip(const CoreRange& strip, uint32_t num_cores) {
    const CoreRangeSet strip_range_set(strip);
    return corerange_to_cores(strip_range_set, num_cores, /*row_wise=*/true);
}

std::optional<CoreRange> find_tilize_2x2_block_avoiding(const CoreCoordPairSet& avoid, const tt::tt_metal::CoreCoord& worker_grid) {
    constexpr uint32_t kTilizeBlockWidth = 2;
    constexpr uint32_t kTilizeBlockHeight = 2;

    if (worker_grid.x < kTilizeBlockWidth || worker_grid.y < kTilizeBlockHeight) {
        return std::nullopt;
    }

    const uint32_t sy = worker_grid.y - kTilizeBlockHeight;

    // Prefer eastern columns (legacy pool was x=5,6 on WH).
    for (int sx = static_cast<int>(worker_grid.x) - static_cast<int>(kTilizeBlockWidth); sx >= 0; --sx) {
        bool valid = true;
        for (uint32_t dy = 0; dy < kTilizeBlockHeight && valid; ++dy) {
            for (uint32_t dx = 0; dx < kTilizeBlockWidth && valid; ++dx) {
                if (avoid.contains({static_cast<uint32_t>(sx) + dx, sy + dy})) {
                    valid = false;
                }
            }
        }
        if (valid) {
            return CoreRange(
                {static_cast<uint32_t>(sx), sy},
                {static_cast<uint32_t>(sx) + kTilizeBlockWidth - 1, sy + kTilizeBlockHeight - 1});
        }
    }
    return std::nullopt;
}

std::vector<tt::tt_metal::CoreCoord> pick_tilize_cores_from_2x2_legacy_order(const CoreRange& block, uint32_t num_cores) {
    const uint32_t sx = block.start_coord.x;
    const uint32_t sy = block.start_coord.y;

    const std::vector<tt::tt_metal::CoreCoord> legacy_order = {
        tt::tt_metal::CoreCoord(sx + 1, sy + 1),
        tt::tt_metal::CoreCoord(sx + 1, sy),
        tt::tt_metal::CoreCoord(sx, sy + 1),
        tt::tt_metal::CoreCoord(sx, sy),
    };

    TT_FATAL(
        num_cores <= legacy_order.size(),
        "pick_tilize_cores_from_2x2_legacy_order: requested {} cores but legacy 2x2 block only has {}",
        num_cores,
        legacy_order.size());

    return std::vector<tt::tt_metal::CoreCoord>(legacy_order.begin(), legacy_order.begin() + num_cores);
}

uint32_t compute_moe_compute_tilize_num_cores(uint32_t hidden_tiles) {
    uint32_t num_cores = std::min(kMoEComputeMaxTilizeCores, hidden_tiles);
    while (num_cores > 1 && hidden_tiles % num_cores != 0) {
        --num_cores;
    }
    return std::max(1u, num_cores);
}

// All worker-core selection happens in LOGICAL coordinates. worker_core_from_logical_core() returns
// VIRTUAL (translated) coordinates, which are contiguous regardless of harvesting, so a logical
// bounding box always maps to a contiguous NoC multicast rectangle. Selection is therefore
// harvesting-agnostic as long as we (a) bound it by compute_with_storage_grid_size() (which already
// accounts for harvesting) and (b) keep the per-group bounding boxes mutually disjoint.

// Insert every cell of a logical rectangle into an avoid set. Avoiding the whole matmul bounding box
// (not just the matmul cores) is what guarantees the tilize/combine rectangles stay disjoint from it:
// the tilize drain multicasts metadata/data to the matmul bbox rectangle, so any tilize/combine core
// inside that rectangle would be spuriously signalled or have its L1 corrupted.
void add_bbox_cells(CoreCoordPairSet& avoid, const CoreRange& bbox) {
    for (uint32_t y = bbox.start_coord.y; y <= bbox.end_coord.y; ++y) {
        for (uint32_t x = bbox.start_coord.x; x <= bbox.end_coord.x; ++x) {
            avoid.insert({x, y});
        }
    }
}

// Matmul ring placement. The base set is the optimal DRAM-bank -> worker assignment (WH: 12 == ring
// size; BH: 8 banks). When the ring is larger than the bank count (BH N=12/16), pad with extra cores
// INSIDE the base bounding box so the bbox does not grow (keeping room for tilize/combine elsewhere).
// Extras prefer the existing DRAM-adjacent columns (better locality for dm0's weight reads), then any
// free cell inside the bbox; extras route around mux cells when possible.
//
// This builds the PREFERRED (DRAM-bank-adjacent) ring. dm0's weight reads are DRAM-bank-id based
// (get_noc_addr_from_bank_id), so the ring is functionally correct from any cores, but this placement
// gives the best locality. When it collides with mux or leaves no disjoint combine/tilize room, the
// caller relocates matmul to build_compact_matmul_cores() instead (see its note: the compact ring must
// stay column-0-anchored, established experimentally — [3,0]-[9,1] hangs while [0,0]-[9,1] passes).
std::vector<tt::tt_metal::CoreCoord> build_matmul_ring_cores(
    ttnn::MeshDevice* mesh_device, uint32_t ring_size, const CoreCoordPairSet& mux_pairs) {
    auto cores = mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);
    const uint32_t num_banks = static_cast<uint32_t>(cores.size());

    if (num_banks >= ring_size) {
        cores.resize(ring_size);
        return cores;
    }

    CoreCoordPairSet used = core_coords_to_pair_set(cores);
    const CoreRange base_bbox = CoreRangeSet(cores).bounding_box();

    std::set<uint32_t> matmul_cols;
    for (const auto& c : cores) {
        matmul_cols.insert(c.x);
    }

    const auto try_add = [&](uint32_t x, uint32_t y) {
        const std::pair<uint32_t, uint32_t> p{x, y};
        if (!used.contains(p) && !mux_pairs.contains(p)) {
            cores.push_back(tt::tt_metal::CoreCoord(x, y));
            used.insert(p);
        }
    };

    // Pass 1: extend the DRAM-adjacent columns at free rows (round-robin keeps banks balanced).
    for (uint32_t y = base_bbox.start_coord.y; y <= base_bbox.end_coord.y && cores.size() < ring_size; ++y) {
        for (uint32_t x : matmul_cols) {
            if (cores.size() >= ring_size) {
                break;
            }
            try_add(x, y);
        }
    }
    // Pass 2: any remaining free cell inside the base bbox.
    for (uint32_t y = base_bbox.start_coord.y; y <= base_bbox.end_coord.y && cores.size() < ring_size; ++y) {
        for (uint32_t x = base_bbox.start_coord.x; x <= base_bbox.end_coord.x && cores.size() < ring_size; ++x) {
            try_add(x, y);
        }
    }

    return cores;
}

// Column-0-anchored compact matmul placement. Used as a FALLBACK when the DRAM-bank-adjacent ring
// cannot be used: either it collides with user mux cores, or its bounding box spans the grid so no
// disjoint combine/tilize layout exists (the WH ROW-dispatch case, where the DRAM-adjacent workers
// occupy the full compute grid). Packs the ring row-major starting at row `y_start`, reserving the
// eastern combine strip (x_limit) and the top two tilize rows so combine/tilize have room.
//
// `y_start` lets the caller slide the block UP off the bottom rows: a wide combine group (e.g.
// deepseek hidden=7168 -> 16 combine cores) on a short grid (WH 8x9) can only be placed row-major in
// the bottom rows, so the matmul block must vacate them or no disjoint combine bbox exists. The
// caller searches y_start (bottom-first) and keeps the first value that yields a disjoint layout.
//
// CRITICAL (established experimentally, not assumed): the compact block MUST stay anchored at column
// x=0. A compact ring whose bbox starts at x>0 hangs / corrupts on device (observed: logical
// [3,0]-[9,1] hangs, while [0,0]-[9,1] and [0,3]-[9,4] pass). We therefore only use rows whose
// leftmost (x=0) cell is mux-free (skipping a whole row otherwise); mux cells in non-leftmost columns
// are left as gaps inside the bbox (mux inside the matmul bbox is benign, verified).
std::vector<tt::tt_metal::CoreCoord> build_compact_matmul_cores(
    const tt::tt_metal::CoreCoord& worker_grid, uint32_t ring_size, const CoreCoordPairSet& mux_pairs, uint32_t y_start) {
    std::vector<tt::tt_metal::CoreCoord> cores;
    cores.reserve(ring_size);

    const uint32_t x_limit = worker_grid.x > kMoEComputeCombineStripWidth
                                 ? static_cast<uint32_t>(worker_grid.x) - kMoEComputeCombineStripWidth
                                 : static_cast<uint32_t>(worker_grid.x);
    const uint32_t y_limit =
        worker_grid.y > 2 ? static_cast<uint32_t>(worker_grid.y) - 2 : static_cast<uint32_t>(worker_grid.y);

    for (uint32_t y = y_start; y < y_limit && cores.size() < ring_size; ++y) {
        // Anchor at column 0: a row may only contribute if its leftmost cell is mux-free, otherwise
        // the block would start at x>0 (a placement that deadlocks on device).
        if (mux_pairs.contains({0, y})) {
            continue;
        }
        for (uint32_t x = 0; x < x_limit && cores.size() < ring_size; ++x) {
            if (!mux_pairs.contains({x, y})) {
                cores.push_back(tt::tt_metal::CoreCoord(x, y));
            }
        }
    }

    return cores;
}

struct PlacedWorkers {
    std::vector<tt::tt_metal::CoreCoord> combine_cores;
    std::vector<tt::tt_metal::CoreCoord> tilize_cores;
    CoreRange combine_bounding_box;
    CoreRange tilize_bounding_box;
};

// Place combine (dense 2-wide strip) and tilize (2x2 block on the top two rows) so that all three
// bounding boxes (matmul, combine, tilize) are mutually disjoint and avoid the mux region. Returns
// nullopt if no disjoint layout exists for the given matmul bounding box. The caller first tries this
// with the DRAM-adjacent matmul bbox; on nullopt it retries with the compact matmul bbox, and only
// raises a hard error if that also fails. The top two rows are reserved for tilize
// (drain == tilize_cores[0]); combine takes the rows below.
//
// `hidden_tiles` guards a subtle correctness constraint: the program factory converts each tilize
// core's byte subregion back to a tile count via integer division (subtoken_size / tile_width_bytes),
// so the selected tilize count must divide hidden_tiles exactly. target_tilize_num_cores always does
// (it comes from compute_moe_compute_tilize_num_cores), but if a user-supplied mux_core_range_set
// places mux cells in the top rows and blocks the 2x2 block, the fallback can land on a smaller
// value that is not a divisor. The divisibility check below skips those, ensuring the first
// accepted count is always a divisor of hidden_tiles. 1 always divides, so the loop always succeeds.
std::optional<PlacedWorkers> place_combine_and_tilize(
    const tt::tt_metal::CoreCoord& worker_grid,
    const CoreRange& matmul_bounding_box,
    const CoreCoordPairSet& mux_pairs,
    uint32_t num_combine_cores,
    uint32_t target_tilize_num_cores,
    uint32_t hidden_tiles) {
    CoreCoordPairSet base_avoid = mux_pairs;
    add_bbox_cells(base_avoid, matmul_bounding_box);

    const uint32_t combine_max_y =
        worker_grid.y >= 3 ? static_cast<uint32_t>(worker_grid.y) - 3 : static_cast<uint32_t>(worker_grid.y) - 1;
    const uint32_t tilize_min_y =
        worker_grid.y >= 2 ? static_cast<uint32_t>(worker_grid.y) - 2 : static_cast<uint32_t>(worker_grid.y) - 1;

    const uint32_t combine_strip_height =
        (num_combine_cores + kMoEComputeCombineStripWidth - 1) / kMoEComputeCombineStripWidth;

    std::vector<tt::tt_metal::CoreCoord> combine_cores;
    const auto combine_strip_opt =
        find_combine_strip_avoiding(base_avoid, worker_grid, combine_strip_height, combine_max_y);
    if (combine_strip_opt.has_value()) {
        combine_cores = pick_combine_cores_from_strip(combine_strip_opt.value(), num_combine_cores);
    } else {
        combine_cores = pick_worker_cores_row_major_avoiding(base_avoid, worker_grid, num_combine_cores, combine_max_y);
    }
    if (combine_cores.size() != num_combine_cores) {
        return std::nullopt;
    }
    const CoreRange combine_bounding_box = CoreRangeSet(combine_cores).bounding_box();
    // selective_reduce_combine multicasts across combine_bounding_box as a full rectangle.
    // Reject sparse fallback placements so destination count and rectangle stay consistent.
    if (combine_bounding_box.size() != num_combine_cores) {
        return std::nullopt;
    }
    if (combine_bounding_box.intersects(matmul_bounding_box)) {
        return std::nullopt;
    }

    CoreCoordPairSet tilize_avoid = base_avoid;
    add_bbox_cells(tilize_avoid, combine_bounding_box);

    for (uint32_t tilize_num_cores = target_tilize_num_cores; tilize_num_cores >= 1; --tilize_num_cores) {
        if (hidden_tiles % tilize_num_cores != 0) {
            continue;
        }
        std::vector<tt::tt_metal::CoreCoord> tilize_cores;
        const auto tilize_block_opt = find_tilize_2x2_block_avoiding(tilize_avoid, worker_grid);
        if (tilize_block_opt.has_value()) {
            tilize_cores = pick_tilize_cores_from_2x2_legacy_order(tilize_block_opt.value(), tilize_num_cores);
        } else {
            tilize_cores = pick_tilize_cores_in_upper_rows(tilize_avoid, worker_grid, tilize_num_cores, tilize_min_y);
        }
        if (tilize_cores.size() != tilize_num_cores) {
            continue;
        }
        const CoreRange tilize_bounding_box = CoreRangeSet(tilize_cores).bounding_box();
        if (tilize_bounding_box.intersects(combine_bounding_box) ||
            tilize_bounding_box.intersects(matmul_bounding_box)) {
            continue;
        }
        return PlacedWorkers{
            .combine_cores = std::move(combine_cores),
            .tilize_cores = std::move(tilize_cores),
            .combine_bounding_box = combine_bounding_box,
            .tilize_bounding_box = tilize_bounding_box};
    }
    return std::nullopt;
}

}  // namespace

MoEComputeCoreSelection select_moe_compute_cores(
    ttnn::MeshDevice* mesh_device,
    uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set,
    uint32_t bh_ring_size) {
    /*
     * Core-selection strategy (all in LOGICAL coordinates; harvesting is transparent because
     * worker_core_from_logical_core() maps logical -> contiguous VIRTUAL coords for multicast):
     *
     *   1. matmul (preferred): DRAM-bank-adjacent workers (perf), padded inside their bbox for ring
     *      N > banks. Used when it does NOT collide with mux AND leaves room for a disjoint
     *      combine/tilize layout.
     *   2. matmul (fallback): a COMPACT, column-0-anchored ring. Required because:
     *        - On WH ROW dispatch the DRAM-bank-adjacent workers span the full compute grid, so the
     *          matmul bbox becomes the entire grid and no disjoint combine/tilize layout exists ->
     *          hang/crash. Compacting the ring shrinks its bbox so tilize/combine fit.
     *        - When user mux cores land on the DRAM-adjacent ring, we relocate matmul instead of the
     *          ring (the ring is what the user can query and route mux around).
     *      The block's vertical offset is searched bottom-first: on short grids a wide combine group
     *      (e.g. WH 8x9 + deepseek's 16 combine cores) can only be placed row-major in the bottom
     *      rows, so the matmul block slides up to vacate them and keep the combine bbox disjoint.
     *   3. combine + tilize: placed in the region NOT spanned by the (chosen) matmul bounding box,
     *      avoiding the mux region, as dense rectangles with mutually disjoint bounding boxes.
     *
     * Why disjoint bounding boxes are required among the THREE worker groups (from the kernels, not
     * comments): the tilize drain multicasts metadata/data/counts to the tilize bbox and
     * metadata_ready to the matmul bbox (metadata_ready is shared on tilize ∪ matmul). The combine
     * reader/writer multicast to the combine bbox. So a *worker* core of one group sitting inside
     * another group's bbox is spuriously signalled / has its L1 corrupted. tilize_cores[0] is the
     * drain; tilize_cores[T/2] the secondary mcaster.
     *
     * CRITICAL constraint on the compact ring (established experimentally, not assumed): it MUST stay
     * anchored at column x=0. A compact ring whose bbox starts at x>0 deadlocks / corrupts on device
     * (observed: logical [3,0]-[9,1] hangs, while [0,0]-[9,1] and [0,3]-[9,4] pass). build_compact_*
     * therefore only uses rows whose leftmost (x=0) cell is mux-free.
     *
     * MUX cores (user-provided, run tt_fabric_mux in the separate combine program): the only enforced
     * constraint is CORE-LEVEL disjointness from every worker core (two concurrent programs cannot
     * share a core's RISCs). Mux is NOT required to be outside the worker bounding boxes — the moe
     * per-expert/metadata multicasts that geometrically cover mux cells inside the matmul/all-worker
     * bbox are benign in practice (verified: mux={(1,1)-(3,3)} and {(0,4)-(2,6)} lie inside the BH
     * matmul bbox x=0..7 and pass). So mux is treated as an avoid-set of cells: every worker group
     * (matmul ring, tilize, combine) is placed to dodge the mux cells, and combine/tilize are placed
     * after the matmul ring so they also route around the matmul bbox.
     */
    constexpr uint32_t tile_width = 32;
    const uint32_t hidden_tiles = hidden_size / tile_width;

    const uint32_t ring_size = (mesh_device->arch() == tt::ARCH::BLACKHOLE) ? bh_ring_size : 12u;
    if (mesh_device->arch() == tt::ARCH::BLACKHOLE) {
        TT_FATAL(
            ring_size == 8 || ring_size == 12 || ring_size == 16,
            "moe_compute: unsupported BH ring size N={}, supported values are {{8, 12, 16}}",
            ring_size);
    }

    const tt::tt_metal::CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const uint32_t num_combine_cores = combine_token_parallel_cores * combine_data_parallel_cores;
    const uint32_t target_tilize_num_cores = compute_moe_compute_tilize_num_cores(hidden_tiles);

    CoreCoordPairSet mux_pairs;
    if (!mux_core_range_set.empty()) {
        mux_pairs = core_coords_to_pair_set(corerange_to_cores(mux_core_range_set));
    }

    // Prefer the DRAM-bank-adjacent ring; fall back to a column-0-anchored compact ring when it
    // collides with mux or leaves no disjoint combine/tilize room (WH ROW dispatch spans the grid).
    std::vector<tt::tt_metal::CoreCoord> matmul_cores;
    std::optional<PlacedWorkers> placed;
    bool used_compact_matmul = false;

    const std::vector<tt::tt_metal::CoreCoord> dram_ring = build_matmul_ring_cores(mesh_device, ring_size, mux_pairs);
    bool dram_ring_hits_mux = false;
    for (const auto& c : dram_ring) {
        if (mux_pairs.contains({c.x, c.y})) {
            dram_ring_hits_mux = true;
            break;
        }
    }
    if (dram_ring.size() == ring_size && !dram_ring_hits_mux) {
        placed = place_combine_and_tilize(
            worker_grid,
            CoreRangeSet(dram_ring).bounding_box(),
            mux_pairs,
            num_combine_cores,
            target_tilize_num_cores,
            hidden_tiles);
        if (placed.has_value()) {
            matmul_cores = dram_ring;
        }
    }

    if (!placed.has_value()) {
        // Search the compact block's vertical offset (column-0-anchored throughout). Bottom-first
        // (y_start=0) reproduces the validated BH layout; sliding the block up frees the bottom rows
        // for a wide row-major combine group on short grids (WH 8x9 + deepseek's 16 combine cores),
        // which is the only way to get a combine bbox disjoint from the matmul bbox there. Keep the
        // first y_start that yields a fully-disjoint combine/tilize layout.
        std::vector<tt::tt_metal::CoreCoord> compact_ring;
        for (uint32_t y_start = 0; y_start < static_cast<uint32_t>(worker_grid.y); ++y_start) {
            std::vector<tt::tt_metal::CoreCoord> candidate = build_compact_matmul_cores(worker_grid, ring_size, mux_pairs, y_start);
            if (candidate.size() != ring_size) {
                // Higher y_start can only fit fewer cores, so no point continuing.
                break;
            }
            std::optional<PlacedWorkers> candidate_placed = place_combine_and_tilize(
                worker_grid,
                CoreRangeSet(candidate).bounding_box(),
                mux_pairs,
                num_combine_cores,
                target_tilize_num_cores,
                hidden_tiles);
            if (candidate_placed.has_value()) {
                compact_ring = std::move(candidate);
                placed = std::move(candidate_placed);
                break;
            }
        }
        TT_FATAL(
            placed.has_value(),
            "moe_compute: could not place a column-0-anchored compact {}-core matmul ring together with {} "
            "combine + {} tilize cores disjointly, avoiding {} mux cores, on the {}x{} worker grid",
            ring_size,
            num_combine_cores,
            target_tilize_num_cores,
            mux_core_range_set.num_cores(),
            worker_grid.x,
            worker_grid.y);
        matmul_cores = std::move(compact_ring);
        used_compact_matmul = true;
    }

    std::vector<tt::tt_metal::CoreCoord> combine_cores = std::move(placed->combine_cores);
    std::vector<tt::tt_metal::CoreCoord> tilize_cores = std::move(placed->tilize_cores);
    const CoreRange combine_bounding_box = placed->combine_bounding_box;
    const CoreRange tilize_bounding_box = placed->tilize_bounding_box;

    const CoreRangeSet matmul_core_range_set = CoreRangeSet(matmul_cores);
    const CoreRange matmul_bounding_box = matmul_core_range_set.bounding_box();
    const CoreRangeSet tilize_core_range_set = CoreRangeSet(tilize_cores);
    const CoreRangeSet combine_core_range_set = CoreRangeSet(combine_cores);

    // Invariant: the three multicast rectangles must be mutually disjoint. Promoted to hard asserts so
    // a bad layout fails loudly at program-build time instead of hanging on device.
    TT_FATAL(!tilize_bounding_box.intersects(combine_bounding_box), "combine and tilize bounding boxes cannot overlap");
    TT_FATAL(!tilize_bounding_box.intersects(matmul_bounding_box), "tilize and matmul bounding boxes cannot overlap");
    TT_FATAL(!combine_bounding_box.intersects(matmul_bounding_box), "combine and matmul bounding boxes cannot overlap");

    // Hard guard: every worker group is placed around mux (matmul: DRAM ring only used when it does
    // not hit mux, else compact ring skips mux cells; combine/tilize routed around mux), so this should
    // always hold — assert defensively so a bad layout fails loudly instead of hanging on device.
    if (!mux_pairs.empty()) {
        const auto assert_disjoint_from_mux = [&](const std::vector<tt::tt_metal::CoreCoord>& cs, const char* group) {
            for (const auto& c : cs) {
                TT_FATAL(
                    !mux_pairs.contains({c.x, c.y}),
                    "moe_compute: {} core {} overlaps a user mux core; worker placement must avoid mux cores",
                    group,
                    c.str());
            }
        };
        assert_disjoint_from_mux(matmul_cores, "matmul");
        assert_disjoint_from_mux(combine_cores, "combine");
        assert_disjoint_from_mux(tilize_cores, "tilize");
    }

    log_debug(
        tt::LogOp,
        "moe_compute: placement matmul_mode={} mux={} | matmul={} bbox={} | combine={} bbox={} | tilize={} bbox={}",
        used_compact_matmul ? "compact" : "dram-adjacent",
        mux_core_range_set.empty() ? "none" : mux_core_range_set.str(),
        matmul_cores.size(),
        matmul_bounding_box.str(),
        combine_cores.size(),
        combine_bounding_box.str(),
        tilize_cores.size(),
        tilize_bounding_box.str());

    const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(matmul_core_range_set);

    // Stable x-major order matches the combine core indexing used by dm1's OUTPUT_SHARD_CORE_MAP.
    std::sort(combine_cores.begin(), combine_cores.end(), [](const auto& a, const auto& b) {
        return (a.x != b.x) ? a.x < b.x : a.y < b.y;
    });

    const CoreRangeSet combine_matmul_core_range_set = combine_core_range_set.merge(matmul_core_range_set);
    const CoreRangeSet all_worker_cores_range_set = tilize_matmul_core_range_set.merge(combine_core_range_set);

    return {
        .tilize_cores = std::move(tilize_cores),
        .matmul_cores = matmul_cores,
        .tilize_core_range_set = tilize_core_range_set,
        .matmul_core_range_set = matmul_core_range_set,
        .tilize_matmul_core_range_set = tilize_matmul_core_range_set,
        .combine_core_range_set = combine_core_range_set,
        .combine_matmul_core_range_set = combine_matmul_core_range_set,
        .all_worker_cores_range_set = all_worker_cores_range_set,
        .combine_cores = std::move(combine_cores),
        .tilize_bounding_box = tilize_bounding_box,
        .matmul_bounding_box = matmul_bounding_box,
    };
}

}  // namespace ttnn::operations::ccl::common
