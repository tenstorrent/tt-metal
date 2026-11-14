// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>

namespace ttnn::prim {

// Fewest tile columns a core scans when a row is split over several cores.
inline constexpr uint32_t ARGMAX_TILE_MIN_COLS_PER_CORE = 4;

/**
 * How the TILE multi-core argmax splits tile_rows x tile_cols tiles over num_cores cores.
 *
 * With at least as many tile rows as cores, or rows too narrow to split, every core is its own
 * group and takes whole rows. Otherwise each tile row gets its own group of cores_per_group cores
 * (at most num_cores / tile_rows, at least ARGMAX_TILE_MIN_COLS_PER_CORE columns each).
 */
struct ArgmaxTileMultiCoreSplit {
    uint32_t cores_per_group = 1;
    uint32_t num_groups = 1;
    uint32_t rows_per_group = 1;  // largest row count of any group

    // [begin, count) of the tile rows of group g: balanced, the first groups take one more.
    std::pair<uint32_t, uint32_t> rows_of_group(uint32_t g, uint32_t tile_rows) const {
        const uint32_t base = tile_rows / num_groups;
        const uint32_t extra = tile_rows % num_groups;
        const uint32_t begin = (g * base) + std::min(g, extra);
        return {begin, base + (g < extra ? 1 : 0)};
    }

    // [begin, count) of the tile columns of core c of a group: balanced, the first cores take one more.
    std::pair<uint32_t, uint32_t> cols_of_core(uint32_t c, uint32_t tile_cols) const {
        const uint32_t base = tile_cols / cores_per_group;
        const uint32_t extra = tile_cols % cores_per_group;
        const uint32_t begin = (c * base) + std::min(c, extra);
        return {begin, base + (c < extra ? 1 : 0)};
    }
};

inline ArgmaxTileMultiCoreSplit argmax_tile_multi_core_split(
    uint32_t tile_rows, uint32_t tile_cols, uint32_t num_cores) {
    ArgmaxTileMultiCoreSplit split;
    num_cores = std::max<uint32_t>(num_cores, 1);
    tile_rows = std::max<uint32_t>(tile_rows, 1);
    const uint32_t cores_per_group = std::min(num_cores / tile_rows, tile_cols / ARGMAX_TILE_MIN_COLS_PER_CORE);
    if (cores_per_group < 2) {
        split.num_groups = std::min(tile_rows, num_cores);
        split.rows_per_group = (tile_rows + split.num_groups - 1) / split.num_groups;
        return split;
    }
    // cores_per_group <= num_cores / tile_rows, so every tile row gets a group of its own.
    split.cores_per_group = cores_per_group;
    split.num_groups = tile_rows;
    split.rows_per_group = 1;
    return split;
}

}  // namespace ttnn::prim
