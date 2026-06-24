// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_compile_time_arg_val(0);

    transpose_wh_init(tt::CBIndex::c_24, tt::CBIndex::c_17);

    constexpr uint32_t cb_im0 = tt::CBIndex::c_24;
    constexpr uint32_t cb_out1 = tt::CBIndex::c_17;

    CircularBuffer cb_im0_obj(cb_im0);
    CircularBuffer cb_out1_obj(cb_out1);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < num_tiles; n++) {
        cb_im0_obj.wait_front(1);

        tile_regs_acquire();
        transpose_wh_tile(cb_im0, 0, 0);
        tile_regs_commit();

        cb_im0_obj.pop_front(1);

        cb_out1_obj.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, cb_out1);
        tile_regs_release();

        cb_out1_obj.push_back(1);
    }
}
