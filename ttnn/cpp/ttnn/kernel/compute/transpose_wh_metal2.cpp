// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose_wh.cpp (the legacy copy sits beside this file and still serves the ops on
// the legacy host API).  Compute logic is unchanged; resource access uses Metal 2.0 named handles:
//   - c_0 / c_16 (legacy hard-coded circular-buffer indices) -> dfb::in / dfb::out
//   - NHtWt (legacy positional compile-time arg 0)            -> named compile-time arg NHtWt

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);

    constexpr uint32_t NHtWt = get_arg(args::NHtWt);

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
