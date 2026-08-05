// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"

#include "api/debug/dprint.h"  // [#48552 DIAG - remove after]

// [#48552 DIAG] Block-granularity datacopy standing in for the tilize LLK.
//
// The DFB rhythm here is deliberately identical to compute_kernel_lib::tilize with
// WaitMode::WaitBlock (tilize_helpers.inl): one wait_front/reserve_back/push_back/pop_front
// per block of block_width_tiles entries, NOT per tile. Only the math differs (copy_tile
// instead of tilize_block). A failure here therefore implicates block-granularity DFB
// credit/pointer handling; a pass leaves the tilize LLK + tilize_init as the suspect.
//
// DST holds one tile at a time so wide blocks (block_width_tiles > DST capacity) are safe;
// the block's credits stay held across the inner loop, which is the property under test.
void kernel_main() {
    constexpr auto per_core_block_cnt = get_arg(args::per_core_block_cnt);
    constexpr auto per_core_block_tile_cnt = get_arg(args::per_core_block_tile_cnt);

    unary_op_init_common(dfb::in, dfb::out);
    copy_tile_init(dfb::in);

    DataflowBuffer cb_in(dfb::in);
    DataflowBuffer cb_out(dfb::out);

    DPRINT(
        "[TLZC] BLOCK-DATACOPY entry blocks={} tiles/blk={}\n",
        (uint32_t)per_core_block_cnt,
        (uint32_t)per_core_block_tile_cnt);

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_in.wait_front(per_core_block_tile_cnt);
        cb_out.reserve_back(per_core_block_tile_cnt);

        for (uint32_t i = 0; i < per_core_block_tile_cnt; ++i) {
            tile_regs_acquire();
            copy_tile(dfb::in, i, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, dfb::out);
            tile_regs_release();
        }

        cb_out.push_back(per_core_block_tile_cnt);
        cb_in.pop_front(per_core_block_tile_cnt);
    }

    DPRINT("[TLZC] BLOCK-DATACOPY DONE\n");
}
