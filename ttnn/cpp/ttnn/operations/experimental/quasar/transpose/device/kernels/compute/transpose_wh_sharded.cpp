// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local compute kernel for transpose's sharded WH factory. Resource bindings use the
// Metal 2.0 namespaces (dfb::/args::).

#include <cstdint>

#include "api/compute/transpose_wh.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t NHtWt = get_arg(args::NHtWt);
    uint32_t HtWt = get_arg(args::HtWt);
    uint32_t N = get_arg(args::N);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t Wt = get_arg(args::Wt);

    transpose_wh_init(dfb::cb_in0, dfb::cb_out0);

    DataflowBuffer cb_in(dfb::cb_in0);
    DataflowBuffer cb_out(dfb::cb_out0);

    // transpose a row-major block:
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile

    uint32_t tile_idx = 0;
    uint32_t tile_idx_N = 0;

    cb_in.wait_front(NHtWt);
    cb_out.reserve_back(NHtWt);
    for (uint32_t n = 0; n < N; ++n) {
        tile_idx = tile_idx_N;
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t h = 0; h < Ht; ++h) {
                tile_regs_acquire();
                transpose_wh_tile(dfb::cb_in0, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, dfb::cb_out0);
                tile_regs_release();
                tile_idx += Wt;
            }
            tile_idx = tile_idx - HtWt + 1;
        }
        tile_idx_N += HtWt;
    }
    cb_out.push_back(NHtWt);
    cb_in.pop_front(NHtWt);
}
