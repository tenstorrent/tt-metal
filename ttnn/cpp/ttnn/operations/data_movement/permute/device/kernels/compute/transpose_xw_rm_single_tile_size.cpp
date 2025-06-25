// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"
#include "compute_kernel_api/pack_untilize.h"

namespace NAMESPACE {
void MAIN {
    constexpr uint32_t x_block_size = get_compile_time_arg_val(0);
    constexpr uint32_t w_block_size = get_compile_time_arg_val(1);

    uint32_t num_blocks = get_arg_val<uint32_t>(0);

    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_tilize = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t n = 0; n < num_blocks; n++) {
        // tilize input via unpack and then pack
        tilize_init(cb_in, 1, cb_tilize);

        cb_wait_front(cb_in, x_block_size);
        cb_reserve_back(cb_tilize, 1);

        tilize_block(cb_in, 1, cb_tilize);  // tilize and pack into cb_tilize

        cb_push_back(cb_tilize, 1);
        cb_pop_front(cb_in, x_block_size);

        tilize_uninit(cb_in, cb_tilize);

        // transpose input
        cb_wait_front(cb_tilize, 1);
        transpose_wh_init_short(cb_tilize);
        pack_untilize_dst_init_short<1>(cb_out);

        tile_regs_acquire();
        transpose_wh_tile(cb_tilize, 0, 0);  // transpose call
        tile_regs_commit();

        // pack and untilize
        cb_reserve_back(cb_out, w_block_size);

        tile_regs_wait();
        pack_untilize_dst<1>(cb_out);  // pack call
        tile_regs_release();

        cb_push_back(cb_out, w_block_size);

        pack_untilize_uninit(cb_out);

        cb_pop_front(cb_tilize, 1);
    }
}
}  // namespace NAMESPACE
