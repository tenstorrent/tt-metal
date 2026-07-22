// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr uint32_t per_core_tile_cnt = get_arg(args::per_core_tile_cnt);

    DataflowBuffer cb_in(dfb::in);
    DataflowBuffer cb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    copy_init(dfb::in);
    for (uint32_t b = 0; b < per_core_tile_cnt; ++b) {
        tile_regs_acquire();

        // Pop tile after tile, copy to DST and pack
        cb_in.wait_front(1);
        cb_out.reserve_back(1);
        copy_tile(dfb::in, 0, 0);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);

        cb_in.pop_front(1);
        cb_out.push_back(1);

        tile_regs_release();
    }
}
