// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_unary/trigonometry.h"
#include "compute_kernel_api/mul_int32_sfpu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_compile_time_arg_val(0);
    uint32_t per_core_block_dim = get_compile_time_arg_val(1);

    uint32_t DST_TILE = 7;

    init_sfpu(tt::CBIndex::c_0, tt::CBIndex::c_2);
    for (uint32_t block_index = 0; block_index < per_core_block_cnt; block_index++) {
        cb_reserve_back(tt::CBIndex::c_2, per_core_block_dim);
        for (uint32_t tile_index = 0; tile_index < per_core_block_dim; ++tile_index) {
            tile_regs_acquire();

            // Pop tile after tile, copy to DST and pack
            cb_wait_front(tt::CBIndex::c_0, 1);

            copy_tile(tt::CBIndex::c_0, 0, DST_TILE);

            // #ifdef SFPU_OP_CHAIN_0
            //             SFPU_OP_CHAIN_0
            // #endif
            // (*((volatile uint32_t*)0x18000)) = (uint32_t)0xccccdddd;
            exp_tile_init<true, true, 0x3F800000>();
            exp_tile<true, true>(DST_TILE);

            tile_regs_commit();

            tile_regs_wait();

            pack_tile(DST_TILE, tt::CBIndex::c_2);

            cb_pop_front(tt::CBIndex::c_0, 1);

            tile_regs_release();
        }
        cb_push_back(tt::CBIndex::c_2, per_core_block_dim);
    }
}
}  // namespace NAMESPACE
