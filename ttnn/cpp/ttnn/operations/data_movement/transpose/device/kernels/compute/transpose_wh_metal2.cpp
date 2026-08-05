// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Metal 2.0 fork of transpose_wh.cpp (cross-op shared with legacy permute / nlp_create_qkv_heads /
// split_query_key_value_and_split_heads). The legacy source stays in place, non-Metal-2.0, for its cross-op consumers;
// only the transpose Metal 2.0 factory binds this fork. Sunset when those consumers migrate.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t NHtWt = get_arg(args::NHtWt);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(dfb::in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
