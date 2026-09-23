// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// onepos_pipeline (Perf 2): column sub-blocks of ONE tile-row, shared by reader / compute / writer.
//
// On a one-position walk (the Tensix core owns one tile-row of one column block) the chain
// read -> tilize -> write is strictly serial. Cutting the tile-row into `n` column sub-blocks lets
// compute publish sub-block k's output tiles (so the writer starts on them) while it tilizes k + 1,
// and, with split reads, lets compute start on sub-block k while the reads of k + 1 are landing.
//
// Sub-block k covers tile-columns [first(k), first(k) + width(k)) of the block_width-wide tile-row.
// Every sub-block is at least 2 tiles wide: the fast-tilize unit on a row wider than one tile is 2
// tiles (1-tile blocks are only legal when the whole row is 1 tile, tilize.h WH branch), so a
// trailing 1-tile remainder is merged into the previous sub-block. n == 1 means "not engaged".
// The CB page counts are used as cumulative progress counters: sub-block k is available once
// first(k) + width(k) pages of the tile-row are published (reader) / packed (compute).

#pragma once

#include <cstdint>

namespace onepos {

template <uint32_t block_width, uint32_t nsb_target>
struct SubBlocks {
    static_assert(nsb_target >= 1, "at least one sub-block");
    static constexpr uint32_t sb_raw = (block_width + nsb_target - 1) / nsb_target;
    static constexpr uint32_t sb = sb_raw < 2 ? 2 : sb_raw;
    static constexpr uint32_t n_raw = block_width <= sb ? 1 : (block_width + sb - 1) / sb;
    static constexpr uint32_t rem = block_width - (n_raw - 1) * sb;
    static constexpr uint32_t n = (n_raw > 1 && rem == 1) ? n_raw - 1 : n_raw;  // merge a 1-tile tail
    static constexpr uint32_t first(uint32_t k) { return k * sb; }
    static constexpr uint32_t width(uint32_t k) { return k + 1 < n ? sb : block_width - (n - 1) * sb; }
    // Production order: step p handles sub-block (j0 + p) mod n, where j0 is the sub-block holding
    // the tile the op's writer would write first (store_rows rotates every tile-row's writes to
    // start at column stick_rotation mod valid_width, which spreads the Tensix cores' first writes
    // over all DRAM banks). Reader, compute and writer all derive j0 the same way, so the CB page
    // counts stay cumulative in production order. ONEPOS_NO_ROTATE: j0 = 0 (column order).
    static uint32_t start(uint32_t stick_rotation, uint32_t valid_width) {
#ifdef ONEPOS_NO_ROTATE
        return 0;
#else
        const uint32_t j = (stick_rotation % valid_width) / sb;
        return j < n ? j : n - 1;  // the last sub-block may be wider (merged 1-tile tail)
#endif
    }
    static uint32_t at(uint32_t j0, uint32_t p) { return j0 + p < n ? j0 + p : j0 + p - n; }
};

}  // namespace onepos
