// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <utility>

namespace ttnn::prim::detail {

// Determine the largest subblock size for matmul that:
// 1. Has subblock_h * subblock_w <= dst_size
// 2. subblock_h divides block_height and subblock_w divides block_width
//
// Candidates are ordered by total volume (h*w) descending to maximize
// dest register utilization.
static inline std::pair<uint32_t, uint32_t> determine_largest_subblock_size(
    uint32_t block_height, uint32_t block_width, uint32_t dst_size) {
    constexpr std::array<std::pair<uint32_t, uint32_t>, 20> subblocks = {{
        {2, 4}, {4, 2}, {1, 8}, {8, 1}, {1, 7}, {7, 1}, {2, 3}, {3, 2}, {1, 6}, {6, 1},
        {1, 5}, {5, 1}, {2, 2}, {1, 4}, {4, 1}, {1, 3}, {3, 1}, {1, 2}, {2, 1}, {1, 1},
    }};

    for (auto [subblock_height, subblock_width] : subblocks) {
        if (subblock_height * subblock_width > dst_size) {
            continue;
        }
        if ((block_height % subblock_height != 0) || (block_width % subblock_width != 0)) {
            continue;
        }
        return {subblock_height, subblock_width};
    }
    return {1, 1};
}

// Find the largest granularity value that:
// 1. Is <= max_granularity (typically dst_size or dst_size/2)
// 2. Evenly divides tile_count (so no tiles are dropped in the kernel loop)
//
// This is critical for correctness: the kernel processes dst_tiles per iteration
// for granularity iterations, totaling dst_tiles * granularity tiles. If the
// granularity doesn't divide tile_count, trailing tiles are silently dropped.
static inline uint32_t find_valid_granularity(uint32_t tile_count, uint32_t max_granularity) {
    uint32_t granularity = std::min(tile_count, max_granularity);
    while (granularity > 1 && tile_count % granularity != 0) {
        granularity--;
    }
    return granularity;
}

}  // namespace ttnn::prim::detail
