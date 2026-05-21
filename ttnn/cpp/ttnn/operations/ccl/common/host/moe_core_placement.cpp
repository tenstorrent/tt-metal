// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/ccl/common/host/moe_core_placement.hpp"

#include <algorithm>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

CoreCoordPairSet core_coords_to_pair_set(const std::vector<CoreCoord>& cores) {
    CoreCoordPairSet result;
    for (const auto& core : cores) {
        result.insert({core.x, core.y});
    }
    return result;
}

std::vector<CoreCoord> pick_worker_cores_row_major_avoiding(
    const CoreCoordPairSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores) {
    std::vector<CoreCoord> picked;
    picked.reserve(num_cores);

    for (uint32_t y = 0; y < worker_grid.y && picked.size() < num_cores; ++y) {
        for (uint32_t x = 0; x < worker_grid.x && picked.size() < num_cores; ++x) {
            CoreCoord candidate(x, y);
            if (!avoid.contains({candidate.x, candidate.y})) {
                picked.push_back(candidate);
            }
        }
    }

    return picked;
}

std::vector<CoreCoord> pick_worker_cores_row_major_avoiding(
    const CoreCoordPairSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores, uint32_t max_y_inclusive) {
    std::vector<CoreCoord> picked;
    picked.reserve(num_cores);

    const uint32_t y_limit = std::min(max_y_inclusive + 1, static_cast<uint32_t>(worker_grid.y));
    for (uint32_t y = 0; y < y_limit && picked.size() < num_cores; ++y) {
        for (uint32_t x = 0; x < worker_grid.x && picked.size() < num_cores; ++x) {
            CoreCoord candidate(x, y);
            if (!avoid.contains({candidate.x, candidate.y})) {
                picked.push_back(candidate);
            }
        }
    }

    return picked;
}

std::vector<CoreCoord> pick_tilize_cores_in_upper_rows(
    const CoreCoordPairSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores, uint32_t min_y) {
    std::vector<CoreCoord> picked;
    picked.reserve(num_cores);

    for (int y = static_cast<int>(worker_grid.y) - 1; y >= static_cast<int>(min_y) && picked.size() < num_cores; --y) {
        for (int x = static_cast<int>(worker_grid.x) - 1; x >= 0 && picked.size() < num_cores; --x) {
            CoreCoord candidate(static_cast<uint32_t>(x), static_cast<uint32_t>(y));
            if (!avoid.contains({candidate.x, candidate.y})) {
                picked.push_back(candidate);
            }
        }
    }

    return picked;
}

std::optional<CoreRange> find_combine_strip_avoiding(
    const CoreCoordPairSet& avoid, const CoreCoord& worker_grid, uint32_t strip_height, uint32_t max_y_inclusive) {
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

std::vector<CoreCoord> pick_combine_cores_from_strip(const CoreRange& strip, uint32_t num_cores) {
    const CoreRangeSet strip_range_set(strip);
    return corerange_to_cores(strip_range_set, num_cores, /*row_wise=*/true);
}

std::optional<CoreRange> find_tilize_2x2_block_avoiding(const CoreCoordPairSet& avoid, const CoreCoord& worker_grid) {
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

std::vector<CoreCoord> pick_tilize_cores_from_2x2_legacy_order(const CoreRange& block, uint32_t num_cores) {
    const uint32_t sx = block.start_coord.x;
    const uint32_t sy = block.start_coord.y;

    const std::vector<CoreCoord> legacy_order = {
        CoreCoord(sx + 1, sy + 1),
        CoreCoord(sx + 1, sy),
        CoreCoord(sx, sy + 1),
        CoreCoord(sx, sy),
    };

    TT_FATAL(
        num_cores <= legacy_order.size(),
        "pick_tilize_cores_from_2x2_legacy_order: requested {} cores but legacy 2x2 block only has {}",
        num_cores,
        legacy_order.size());

    return std::vector<CoreCoord>(legacy_order.begin(), legacy_order.begin() + num_cores);
}

CoreCoordPairSet build_moe_compute_avoid_set(
    const CoreCoordPairSet& matmul_avoid_set, const CoreRangeSet& mux_core_range_set) {
    CoreCoordPairSet avoid_set = matmul_avoid_set;
    if (!mux_core_range_set.empty()) {
        const auto mux_pairs = core_coords_to_pair_set(corerange_to_cores(mux_core_range_set));
        avoid_set.insert(mux_pairs.begin(), mux_pairs.end());
    }
    return avoid_set;
}

uint32_t compute_moe_compute_tilize_num_cores(uint32_t hidden_tiles) {
    uint32_t num_cores = std::min(kMoEComputeMaxTilizeCores, hidden_tiles);
    while (num_cores > 1 && hidden_tiles % num_cores != 0) {
        --num_cores;
    }
    return std::max(1u, num_cores);
}

MoEComputeCoreSelection select_moe_compute_cores(
    ttnn::MeshDevice* mesh_device,
    uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set) {
    /*
     * - First tilize core is the drain sync
     * - First ((total_tilize_cores + 1) / 2) tilize cores are primary mcast group
     * - Remaining cores are secondary mcast group (with the first of them being the secondary mcaster)
     */
    constexpr uint32_t tile_width = 32;
    const uint32_t hidden_tiles = hidden_size / tile_width;

    const auto matmul_cores =
        mesh_device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::RISCV_0_default);

    const CoreCoordPairSet matmul_avoid_set = core_coords_to_pair_set(matmul_cores);
    const CoreCoordPairSet placement_avoid_set = build_moe_compute_avoid_set(matmul_avoid_set, mux_core_range_set);
    const CoreCoord worker_grid = mesh_device->compute_with_storage_grid_size();
    const CoreRangeSet matmul_core_range_set = CoreRangeSet(matmul_cores);
    const CoreRange matmul_bounding_box = matmul_core_range_set.bounding_box();

    const uint32_t num_combine_cores = combine_token_parallel_cores * combine_data_parallel_cores;
    const uint32_t combine_strip_height =
        (num_combine_cores + kMoEComputeCombineStripWidth - 1) / kMoEComputeCombineStripWidth;

    // Reserve the top two worker rows for tilize (legacy 2×2 pool) and the remainder for combine.
    const uint32_t combine_max_y =
        worker_grid.y >= 3 ? static_cast<uint32_t>(worker_grid.y) - 3 : static_cast<uint32_t>(worker_grid.y) - 1;
    const uint32_t tilize_min_y =
        worker_grid.y >= 2 ? static_cast<uint32_t>(worker_grid.y) - 2 : static_cast<uint32_t>(worker_grid.y) - 1;

    // Combine placement depends only on placement_avoid_set and worker_grid — compute once.
    std::vector<CoreCoord> combine_cores;
    const auto combine_strip_opt =
        find_combine_strip_avoiding(placement_avoid_set, worker_grid, combine_strip_height, combine_max_y);

    if (combine_strip_opt.has_value()) {
        combine_cores = pick_combine_cores_from_strip(combine_strip_opt.value(), num_combine_cores);
    } else {
        combine_cores =
            pick_worker_cores_row_major_avoiding(placement_avoid_set, worker_grid, num_combine_cores, combine_max_y);
    }

    TT_FATAL(
        combine_cores.size() == num_combine_cores,
        "Could not find {} combine cores on {}x{} worker grid (matmul_cores={}, mux_cores={})",
        num_combine_cores,
        worker_grid.x,
        worker_grid.y,
        matmul_cores.size(),
        mux_core_range_set.num_cores());

    CoreCoordPairSet tilize_avoid_set = placement_avoid_set;
    const auto combine_pairs = core_coords_to_pair_set(combine_cores);
    tilize_avoid_set.insert(combine_pairs.begin(), combine_pairs.end());

    const CoreRange combine_bounding_box = CoreRangeSet(combine_cores).bounding_box();

    // Retry tilize placement with decreasing core count until we find a non-overlapping layout.
    uint32_t target_tilize_num_cores = compute_moe_compute_tilize_num_cores(hidden_tiles);
    std::vector<CoreCoord> tilize_cores;
    CoreRange tilize_bounding_box({0, 0}, {0, 0});
    bool found_placement = false;

    for (uint32_t tilize_num_cores = target_tilize_num_cores; tilize_num_cores >= 1; --tilize_num_cores) {
        const auto tilize_block_opt = find_tilize_2x2_block_avoiding(tilize_avoid_set, worker_grid);
        if (tilize_block_opt.has_value()) {
            tilize_cores = pick_tilize_cores_from_2x2_legacy_order(tilize_block_opt.value(), tilize_num_cores);
        } else {
            tilize_cores =
                pick_tilize_cores_in_upper_rows(tilize_avoid_set, worker_grid, tilize_num_cores, tilize_min_y);
        }
        if (tilize_cores.size() != tilize_num_cores) {
            continue;
        }

        const CoreRange trial_tilize_bounding_box = CoreRangeSet(tilize_cores).bounding_box();
        if (trial_tilize_bounding_box.intersects(combine_bounding_box)) {
            continue;
        }

        tilize_bounding_box = trial_tilize_bounding_box;
        found_placement = true;
        break;
    }

    TT_FATAL(
        found_placement,
        "Could not find moe_compute core placement (combine_cores={}, hidden_tiles={}, matmul_cores={})",
        num_combine_cores,
        hidden_tiles,
        matmul_cores.size());

    const CoreRangeSet tilize_core_range_set = CoreRangeSet(tilize_cores);
    const CoreRangeSet combine_core_range_set = CoreRangeSet(combine_cores);

    log_info(
        tt::LogOp,
        "moe_compute: selected tilize cores {}, combine cores {}, matmul cores {}",
        tilize_cores.size(),
        combine_cores.size(),
        matmul_cores.size());

    const CoreRangeSet tilize_matmul_core_range_set = tilize_core_range_set.merge(matmul_core_range_set);

    // Multicast rectangles for tilize/combine must stay disjoint. Matmul multicast already uses the
    // DRAM-worker bounding box, which can overlap those rectangles geometrically on some grids even
    // when worker cores are disjoint (same as the legacy hardcoded pools).
    TT_FATAL(!tilize_bounding_box.intersects(combine_bounding_box), "combine and tilize bounding boxes cannot overlap");

    // Stable x-major order matches legacy moe_compute combine core indexing.
    std::sort(combine_cores.begin(), combine_cores.end(), [](const auto& a, const auto& b) {
        return (a.x != b.x) ? a.x < b.x : a.y < b.y;
    });

    const CoreRangeSet combine_matmul_core_range_set = combine_core_range_set.merge(matmul_core_range_set);
    const CoreRangeSet all_worker_cores_range_set = tilize_matmul_core_range_set.merge(combine_core_range_set);

    return {
        .tilize_cores = std::move(tilize_cores),
        .matmul_cores = matmul_cores,
        .tilize_core_range_set = std::move(tilize_core_range_set),
        .matmul_core_range_set = std::move(matmul_core_range_set),
        .tilize_matmul_core_range_set = std::move(tilize_matmul_core_range_set),
        .combine_core_range_set = std::move(combine_core_range_set),
        .combine_matmul_core_range_set = std::move(combine_matmul_core_range_set),
        .all_worker_cores_range_set = std::move(all_worker_cores_range_set),
        .combine_cores = std::move(combine_cores),
        .tilize_bounding_box = tilize_bounding_box,
        .matmul_bounding_box = matmul_bounding_box,
    };
}

}  // namespace ttnn::operations::ccl::common
