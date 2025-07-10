// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"
#include "tt_metal/hw/inc/circular_buffer.h"

namespace NAMESPACE {

void MAIN {
    // X = output width
    // Y = output height
    // input shape = (..., H, W)
    // output shape = (..., Y, X)

    /**
     * This kernel takes in the contiguous XW block read in in the reader kernel and transposes is to a WX block, ready
     * to be written out The transpose LLK does not support transposing a tile without faces/subtiles, so we need to
     * rearrange it into its faces, transpose, and then pack it back such that it's de-faced (WX, where X is contiguous
     * and isn't divided into subtiles)
     */
    uint32_t start_block = get_arg_val<uint32_t>(0);
    uint32_t end_block = get_arg_val<uint32_t>(1);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t block = start_block; block < end_block; block++) {
        // tilize input via unpack and then pack
        tilize_init(cb_in, 1, cb_tilize);

        cb_wait_front(cb_in, 1);
        cb_reserve_back(cb_tilize, 1);

        tilize_block(cb_in, 1, cb_tilize);

        cb_push_back(cb_tilize, 1);
        cb_pop_front(cb_in, 1);

        tilize_uninit(cb_in, cb_tilize);

        // transpose input
        cb_wait_front(cb_tilize, 1);

        transpose_wh_init_short(cb_tilize);
        pack_untilize_dst_init_short<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, 1);

        tile_regs_wait();
        pack_untilize_dst<1>(cb_out);  // pack call
        tile_regs_release();

        cb_push_back(cb_out, 1);

        pack_untilize_uninit(cb_out);

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
