// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 op-local compute kernel for transpose's tiled WH factory. The compute logic is
// unchanged; only the resource bindings move to the Metal 2.0 namespaces (dfb::/args::).

#include <cstdint>

#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    std::uint32_t NHtWt = get_arg(args::NHtWt);

    compute_kernel_hw_startup(dfb::cb_in0, dfb::cb_out0);
    transpose_init(dfb::cb_in0);

    DataflowBuffer cb_in(dfb::cb_in0);
    DataflowBuffer cb_out(dfb::cb_out0);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (std::uint32_t n = 0; n < NHtWt; n++) {
        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(dfb::cb_in0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, dfb::cb_out0);
        tile_regs_release();

        cb_out.push_back(1);
        cb_in.pop_front(1);
    }
}
