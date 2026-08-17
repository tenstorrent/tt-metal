// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of transpose_wh.cpp (see the legacy copy alongside).
// Compute logic is unchanged; resource access uses Metal 2.0 named handles:
//   - c_0 / c_16 (legacy magic CB indices) -> dfb::cb_in / dfb::cb_out
//   - NHtWt (legacy RTA 0)                 -> named RTA
// Forked (not modified in place) because the legacy copy is still used by the transpose op.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t NHtWt = get_arg(args::NHtWt);

    constexpr auto cb_in = dfb::cb_in;
    constexpr auto cb_out = dfb::cb_out;

    compute_kernel_hw_startup(cb_in, cb_out);
    transpose_init(cb_in);

    DataflowBuffer dfb_in(cb_in);
    DataflowBuffer dfb_out(cb_out);

    // transpose a row-major block:
    // - assumes the tiles come in in column major order from reader
    // - uses reader_unary_transpose_wh
    // - transpose_wh each tile
    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(cb_in, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
