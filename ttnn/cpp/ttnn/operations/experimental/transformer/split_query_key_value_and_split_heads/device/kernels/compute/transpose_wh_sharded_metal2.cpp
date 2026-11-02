// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);

    DataflowBuffer dfb_in_obj(dfb::in);
    DataflowBuffer dfb_out_obj(dfb::out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < num_tiles; n++) {
        dfb_in_obj.wait_front(1);

        tile_regs_acquire();
        transpose_tile(dfb::in, 0, 0);
        tile_regs_commit();

        dfb_in_obj.pop_front(1);

        dfb_out_obj.reserve_back(1);

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out_obj.push_back(1);
    }
}
