// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/debug/dprint.h"  // [TC-DBG #51270] remove after typecast multi-tile isolation

void kernel_main() {
    constexpr uint32_t per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr uint32_t per_core_block_dim = get_arg(args::per_core_block_dim);

    // dfb::in  — the typecast source pages, filled by this factory's reader
    // dfb::out — the typecast result pages, drained by the writer (or, on the sharded path,
    //            resident in the borrowed output buffer with no writer to drain it)
    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    compute_kernel_hw_startup(dfb::in, dfb::out);
    copy_init(dfb::in);
    // [TC-DBG #51270] per_core_block_cnt distinguishes the two grids: it is 2 on the single-compute-node
    // 1x3 grid (both tiles on one core -> the failing path) and 1 on the 2x3 grid (one tile per core).
    // Unguarded DPRINT (matches the eltwise compute-kernel idiom); may print once per active TRISC.
    DPRINT("TC_CMP cnt={}\n", per_core_block_cnt);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        // [TC-DBG #51270] block marker: confirms the loop runs twice on 1x3 and correlates the per-tile
        // TYPECAST_LLK_INIT below with the tile the writer reports as wrong.
        DPRINT("TC_CMP blk={}\n", block_index);
        dfb_out.reserve_back(per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            dfb_in.wait_front(1);

            copy_tile(dfb::in, 0, 0);

            TYPECAST_LLK_INIT();
            TYPECAST_LLK(0);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(0, dfb::out);

            dfb_in.pop_front(1);

            tile_regs_release();
        }
        dfb_out.push_back(per_core_block_dim);
    }
}
