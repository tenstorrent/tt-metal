// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);


    init_sfpu(tt::CB::c_in0);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CB::c_out0, per_core_block_dim);
        for(uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            acquire_dst();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CB::c_in0, 1);

            copy_tile(tt::CB::c_in0, 0, 0);

            // fill_tile_init();

            // //  0x40b00000u is 5.5
            // //  0x40533333u is 3.3
            // //  0x40000000 is 2
            // fill_tile(0, 0x40000000u);

            pack_tile(0, tt::CB::c_out0);

            cb_pop_front(tt::CB::c_in0, 1);

            release_dst();
        }
        cb_push_back(tt::CB::c_out0, per_core_block_dim);
    }

}
}
