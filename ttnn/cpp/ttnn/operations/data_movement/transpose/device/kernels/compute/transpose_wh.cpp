// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"

void kernel_main() {
    uint32_t NHtWt = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_16);
    transpose_init(tt::CBIndex::c_0);

    DataflowBuffer dfb_in(tt::CBIndex::c_0);
    DataflowBuffer dfb_out(tt::CBIndex::c_16);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(tt::CBIndex::c_0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, tt::CBIndex::c_16);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
