// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Column sub-blocks of one tile-row (Perf 2, onepos_pipeline), shared by compute and the writer.
//
// Compute tilizes a tile-row of block_width tiles as column sub-blocks and pushes each sub-block's
// output pages as soon as it is packed; the writer waits for them cumulatively and writes each
// sub-block at once. On a one-position walk (the Tensix core owns ONE tile-row of one column block)
// read -> tilize -> write otherwise run strictly back to back: the writer idles for the whole
// tilize before its first tile write.
//
// Sub-block k covers tile-columns [first(k), first(k) + width(k)) of the tile-row. Every sub-block is
// at least 2 tiles wide: on a row wider than one tile the WH fast-tilize unit is 2 tiles (1-tile
// units are only legal when the whole row is 1 tile, tilize.h WH branch), so a trailing 1-tile
// remainder is merged into the previous sub-block. n == 1 means "nothing to cut" (block_width <= 3).
//
// Production order: step p handles sub-block (j0 + p) mod n, where j0 holds the tile the op's
// store_rows writes first (it rotates every tile-row's writes to start at column
// stick_rotation mod valid_width, which spreads the Tensix cores' first writes over all DRAM banks).
// Compute and the writer derive j0 identically, so the output CB's page counts stay cumulative in
// production order. Measured: a column-order start put every Tensix core's first sub-block on half
// the DRAM banks (+8.6 % [1,1,32,8192], +13 % [1,1,64,4096]).

#pragma once

#include <cstdint>

namespace tilize_sub_blocks {

template <uint32_t block_width, uint32_t sb_tiles>
struct SubBlocks {
    static constexpr uint32_t sb = sb_tiles < 2 ? 2 : sb_tiles;
    static constexpr uint32_t n_raw = block_width <= sb ? 1 : (block_width + sb - 1) / sb;
    static constexpr uint32_t rem = block_width - (n_raw - 1) * sb;
    static constexpr uint32_t n = (n_raw > 1 && rem == 1) ? n_raw - 1 : n_raw;  // merge a 1-tile tail
    static constexpr uint32_t first(uint32_t k) { return k * sb; }
    static constexpr uint32_t width(uint32_t k) { return k + 1 < n ? sb : block_width - (n - 1) * sb; }
    // The sub-block holding the rotated first tile (valid_width: the tile-row's written columns).
    static uint32_t start(uint32_t stick_rotation, uint32_t valid_width) {
        const uint32_t j = (stick_rotation % valid_width) / sb;
        return j < n ? j : n - 1;  // the last sub-block may be wider (merged 1-tile tail)
    }
    static uint32_t at(uint32_t j0, uint32_t p) { return j0 + p < n ? j0 + p : j0 + p - n; }
};

}  // namespace tilize_sub_blocks
