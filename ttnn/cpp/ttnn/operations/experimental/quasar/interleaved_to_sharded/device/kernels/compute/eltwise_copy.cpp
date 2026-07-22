// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_buffer.h"

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Per-core tile count. (Legacy read this as a compile-time arg, but the host emitted it
    // per-core via runtime args; in the typed model it is a named runtime argument.)
    uint32_t per_core_tile_cnt = get_arg(args::num_units);

    compute_kernel_hw_startup(dfb::in0, dfb::out);
    copy_init(dfb::in0);
    copy_init(dfb::in0);

    DataflowBuffer cb_in0(dfb::in0);
    DataflowBuffer cb_out(dfb::out);

    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_in0.wait_front(1);
        cb_out.reserve_back(1);
        copy_tile(dfb::in0, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);

        cb_in0.pop_front(1);
        cb_out.push_back(1);

        tile_regs_release();
    }
}
