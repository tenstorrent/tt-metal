// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    constexpr auto num_tiles = get_arg(args::num_tiles);
    DataflowBuffer src_dfb(dfb::src);
    DataflowBuffer dst_dfb(dfb::dst);
    unary_op_init_common(dfb::src, dfb::dst);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        src_dfb.wait_front(1);
        tile_regs_acquire();
        copy_tile(dfb::src, 0, 0);
        tile_regs_commit();
        src_dfb.pop_front(1);

        dst_dfb.reserve_back(1);
        tile_regs_wait();
        pack_tile(0, dfb::dst, 0);
        tile_regs_release();
        dst_dfb.push_back(1);
    }
}
