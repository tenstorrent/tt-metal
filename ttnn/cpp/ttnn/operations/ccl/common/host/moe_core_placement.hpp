// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <set>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include "ttnn/distributed/types.hpp"

namespace ttnn::operations::ccl::common {

struct CoreCoordSet {
    std::set<std::pair<uint32_t, uint32_t>> coords;

    void insert(const CoreCoord& core);
    bool contains(const CoreCoord& core) const;
    void merge(const CoreCoordSet& other);

    static CoreCoordSet from_cores(const std::vector<CoreCoord>& cores);
};

// Top-right → bottom-left scan (matches moe_gpt tilize placement).
std::vector<CoreCoord> pick_worker_cores_avoiding(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores);

// Row-major scan for first valid W×H rectangle (matches moe_gpt combine placement).
std::optional<CoreRange> find_worker_rectangle_avoiding(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t width, uint32_t height);

// Bottom-left → top-right row-major scan (matches legacy moe_compute combine pool order).
std::vector<CoreCoord> pick_worker_cores_row_major_avoiding(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores);

std::vector<CoreCoord> pick_worker_cores_row_major_avoiding(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores, uint32_t max_y_inclusive);

// Top-right scan while keeping the running tilize bounding box disjoint from matmul_bbox.
std::vector<CoreCoord> pick_tilize_cores_avoiding_matmul_bbox(
    const CoreCoordSet& matmul_avoid,
    const CoreRange& matmul_bounding_box,
    const CoreCoord& worker_grid,
    uint32_t num_cores);

// Top-right scan restricted to y >= min_y (upper worker rows reserved for tilize).
std::vector<CoreCoord> pick_tilize_cores_in_upper_rows(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t num_cores, uint32_t min_y);

constexpr uint32_t kMoEComputeMaxTilizeCores = 4;
// Legacy moe_compute combine pool was 2 columns wide (e.g. x=5,6 on WH).
constexpr uint32_t kMoEComputeCombineStripWidth = 2;

uint32_t compute_moe_compute_tilize_num_cores(uint32_t hidden_tiles);

// Find a 2-column worker strip for combine (prefers eastern columns), matching legacy pool geometry.
std::optional<CoreRange> find_combine_strip_avoiding(
    const CoreCoordSet& avoid, const CoreCoord& worker_grid, uint32_t strip_height, uint32_t max_y_inclusive);

// First N cores from strip in row-wise corerange order (before x-major sort).
std::vector<CoreCoord> pick_combine_cores_from_strip(const CoreRange& strip, uint32_t num_cores);

// Legacy moe_compute tilize pool was a 2×2 block on the top two worker rows (e.g. (5,8)-(6,9) on WH).
std::optional<CoreRange> find_tilize_2x2_block_avoiding(const CoreCoordSet& avoid, const CoreCoord& worker_grid);

// Drain core is index 0: (sx+1, sy+1), then (sx+1, sy), (sx, sy+1), (sx, sy) for block (sx,sy)-(sx+1,sy+1).
std::vector<CoreCoord> pick_tilize_cores_from_2x2_legacy_order(const CoreRange& block, uint32_t num_cores);

struct MoEComputeCoreSelection {
    std::vector<CoreCoord> tilize_cores;
    std::vector<CoreCoord> matmul_cores;
    CoreRangeSet tilize_core_range_set;
    CoreRangeSet matmul_core_range_set;
    CoreRangeSet tilize_matmul_core_range_set;
    CoreRangeSet combine_core_range_set;
    CoreRangeSet combine_matmul_core_range_set;
    CoreRangeSet all_worker_cores_range_set;
    std::vector<CoreCoord> combine_cores;
    CoreRange tilize_bounding_box;
    CoreRange matmul_bounding_box;
};

MoEComputeCoreSelection select_moe_compute_cores(
    ttnn::MeshDevice* mesh_device,
    uint32_t combine_token_parallel_cores,
    uint32_t combine_data_parallel_cores,
    uint32_t hidden_size,
    const CoreRangeSet& mux_core_range_set = CoreRangeSet());

}  // namespace ttnn::operations::ccl::common
