// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);

    transpose_wh_init(tt::CBIndex::c_0, tt::CBIndex::c_16);

    experimental::CircularBuffer cb_in(tt::CBIndex::c_0);
    experimental::CircularBuffer cb_out(tt::CBIndex::c_16);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_wh_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        cb_out.push_back(1);
        cb_in.pop_front(1);
    }
}
