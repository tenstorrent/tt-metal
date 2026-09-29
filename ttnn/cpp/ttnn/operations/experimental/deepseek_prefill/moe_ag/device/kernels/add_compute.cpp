// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Row add, compute: c_16 = c_0 + c_1, tile by tile (row-major data: element-wise ops are layout agnostic).
// CT: 0 TILES (per row)   Common RT: 0 rows, 1 rows per core, 2 grid x
#include <cstdint>
#include "api/compute/common.h"
#include "core_range.hpp"
#include "api/compute/eltwise_binary.h"

void kernel_main() {
    constexpr uint32_t TILES = get_compile_time_arg_val(0);
    constexpr uint32_t DST = 4;  // fp32 DEST, half sync
    const uint32_t n =
        core_range(get_common_arg_val<uint32_t>(0), get_common_arg_val<uint32_t>(1), get_common_arg_val<uint32_t>(2)).n;
    binary_op_init_common(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);
    add_tiles_init(tt::CBIndex::c_0, tt::CBIndex::c_1);
    for (uint32_t i = 0; i < n; ++i) {
        cb_wait_front(tt::CBIndex::c_0, TILES);
        cb_wait_front(tt::CBIndex::c_1, TILES);
        cb_reserve_back(tt::CBIndex::c_16, TILES);
        for (uint32_t j0 = 0; j0 < TILES; j0 += DST) {  // DEST holds DST fp32 tiles
            const uint32_t m = TILES - j0 < DST ? TILES - j0 : DST;
            tile_regs_acquire();
            for (uint32_t j = 0; j < m; ++j) {
                add_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, j0 + j, j0 + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < m; ++j) {
                pack_tile(j, tt::CBIndex::c_16);
            }
            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_16, TILES);
        cb_pop_front(tt::CBIndex::c_0, TILES);
        cb_pop_front(tt::CBIndex::c_1, TILES);
    }
}
