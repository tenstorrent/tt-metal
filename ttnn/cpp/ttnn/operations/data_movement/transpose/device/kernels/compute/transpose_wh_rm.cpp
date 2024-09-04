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
    constexpr uint32_t Ht = get_compile_time_arg_val(0);
    constexpr uint32_t Wt = get_compile_time_arg_val(1);
    constexpr uint32_t HtWt = get_compile_time_arg_val(2);
    #ifdef SHARDED
    constexpr uint32_t num_hw_blocks_per_core = get_compile_time_arg_val(3);
    #else
    uint32_t num_hw_blocks_per_core = get_arg_val<uint32_t>(0);
    #endif

    #ifdef SHARDED
    constexpr auto cb_in = tt::CB::c_intermed0;
    constexpr auto cb_tilize = tt::CB::c_intermed1;
    constexpr auto cb_untilize = tt::CB::c_intermed2;
    constexpr auto cb_out = tt::CB::c_intermed3;
    #else
    constexpr auto cb_in = tt::CB::c_in0;
    constexpr auto cb_tilize = tt::CB::c_intermed0;
    constexpr auto cb_untilize = tt::CB::c_intermed1;
    constexpr auto cb_out = tt::CB::c_out0;
    #endif

    unary_op_init_common(cb_in, cb_out);

    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        // tilize input
        tilize_init_short(cb_in, Wt);
        for (uint32_t h = 0; h < Ht; ++h) {
            cb_wait_front(cb_in, Wt);
            cb_reserve_back(cb_tilize, Wt);
            tilize_block(cb_in, Wt, cb_tilize);
            cb_push_back(cb_tilize, Wt);
            cb_pop_front(cb_in, Wt);
        }
        tilize_uninit(cb_in);

        // transpose
        cb_wait_front(cb_tilize, HtWt);
        uint32_t tile_idx = 0;
        if constexpr(Ht > 8) { // cannot do pack_untilize since dst regs won't fit
            for (uint32_t w = 0; w < Wt; ++w) {
                transpose_wh_init_short(cb_tilize);
                cb_reserve_back(cb_untilize, Ht);
                for (uint32_t h = 0; h < Ht; ++h) {
                    tile_regs_acquire();
                    transpose_wh_tile(cb_tilize, tile_idx, 0);
                    tile_regs_commit();
                    tile_regs_wait();
                    pack_tile(0, cb_untilize);
                    tile_regs_release();
                    tile_idx += Wt;
                }
                tile_idx = tile_idx - HtWt + 1;
                cb_push_back(cb_untilize, Ht);

                // tilize
                // need to add this hw config here, otherwise pcc is bad
                UNPACK(( llk_unpack_untilize_hw_configure_disaggregated<DST_ACCUM_MODE>(cb_untilize) ));
                untilize_init_short(cb_untilize);
                cb_wait_front(cb_untilize, Ht);
                cb_reserve_back(cb_out, Ht);
                untilize_block(cb_untilize, Ht, cb_out);
                cb_push_back(cb_out, Ht);
                cb_pop_front(cb_untilize, Ht);
                untilize_uninit(cb_untilize);
            }

        } else {

            transpose_wh_init_short(cb_tilize);
            for (uint32_t w = 0; w < Wt; ++w) {
                tile_regs_acquire();
                for (uint32_t h = 0; h < Ht; ++h) {
                    transpose_wh_tile(cb_tilize, tile_idx, h);
                    tile_idx += Wt;
                }
                tile_regs_commit();

                pack_untilize_dst_init_short<Ht>(cb_out);
                cb_reserve_back(cb_out, Ht);
                tile_regs_wait();
                pack_untilize_dst<Ht>(cb_out);
                pack_untilize_uninit();
                tile_regs_release();
                cb_push_back(cb_out, Ht);

                cb_wait_front(cb_out, Ht);
                tile_idx = tile_idx - HtWt + 1;
            }
        }
        cb_pop_front(cb_tilize, HtWt);

    }

}
}
