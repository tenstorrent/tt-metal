// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/tilize.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"
#include "api/debug/dprint_tensix.h"

// #include "api/debug/dprint.h"

void print_tile(
    uint8_t cb,
    int tile,
    uint8_t max_h,
    uint8_t max_w,
    const char* info,
    bool endl_rows = true,
    bool print_untilized = false) {
    DPRINT << "++ Tile " << static_cast<int>(cb) << ':' << tile << ' ' << info << ENDL();
    for (uint8_t r = 0; r < max_h; r++) {
        const auto sr = SliceRange{.h0 = r, .h1 = static_cast<uint8_t>(r + 1), .hs = 1, .w0 = 0, .w1 = max_w, .ws = 1};
        DPRINT << static_cast<int>(r) << ": " << TileSlice<64>(cb, tile, sr, endl_rows, print_untilized) << ENDL();
    }
}

void kernel_main() {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_tile_cnt = get_compile_time_arg_val(1);
    UNPACK(DPRINT << "blk/c=" << uint32_t(per_core_block_cnt) << " tle/blk=" << per_core_block_tile_cnt << ENDL());
    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    tilize_init(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

    for (uint32_t b = 0; b < per_core_block_cnt; ++b) {
        cb_wait_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
        UNPACK(print_tile(tt::CBIndex::c_0, 0, 32, 32, "in"); print_tile(tt::CBIndex::c_0, 1, 32, 32, "in"););
        cb_reserve_back(tt::CBIndex::c_16, per_core_block_tile_cnt);

        tilize_block(tt::CBIndex::c_0, per_core_block_tile_cnt, tt::CBIndex::c_16);

        PACK(print_tile(tt::CBIndex::c_16, 0, 32, 32, "out"); print_tile(tt::CBIndex::c_16, 1, 32, 32, "out"));
        cb_push_back(tt::CBIndex::c_16, per_core_block_tile_cnt);
        cb_pop_front(tt::CBIndex::c_0, per_core_block_tile_cnt);
    }
}
