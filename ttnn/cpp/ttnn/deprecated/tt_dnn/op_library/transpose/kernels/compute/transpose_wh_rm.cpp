// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/transpose_wh.h"
#include "compute_kernel_api/tilize.h"
#include "compute_kernel_api/untilize.h"


namespace NAMESPACE {
void MAIN {

    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(0);

    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);

    constexpr auto cb_in0 = tt::CB::c_in0;
    constexpr auto cb_im = tt::CB::c_intermed0;
    constexpr auto cb_im_tp = tt::CB::c_intermed1;
    constexpr auto cb_out0 = tt::CB::c_out0;

    unary_op_init_common(cb_in0, cb_out0);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // tilize input
        tilize_init_short(cb_in0, Wt);
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_wait_front(cb_in0, Wt);
            cb_reserve_back(cb_im, Wt);
            tilize_block(cb_in0, Wt, cb_im);
            cb_push_back(cb_im, Wt);
            cb_pop_front(cb_in0, Wt);
        }
        tilize_uninit(cb_in0);

        // transpose
        transpose_wh_init_short(cb_im);
        cb_wait_front(cb_im, HtWt);
        cb_reserve_back(cb_im_tp, HtWt);
        uint32_t tile_idx = 0;
        for (uint32_t w = 0; w < Wt; ++w) {
            for (uint32_t h = 0; h < Ht; ++h) {
                tile_regs_acquire();
                transpose_wh_tile(cb_im, tile_idx, 0);
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_im_tp);
                tile_regs_release();
                tile_idx += Wt;
            }
            tile_idx = tile_idx - HtWt + 1;
        }
        cb_push_back(cb_im_tp, HtWt);
        cb_pop_front(cb_im, HtWt);

        // untilize
        untilize_init_short(cb_im_tp);
        cb_wait_front(cb_im_tp, HtWt);
        for (uint32_t w = 0; w < Wt; ++w) {
            cb_reserve_back(cb_out0, Ht);
            untilize_block(cb_im_tp, Ht, cb_out0);
            cb_push_back(cb_out0, Ht);
            cb_pop_front(cb_im_tp, Ht);
        }
        untilize_uninit(cb_im_tp);
    }

}
}
