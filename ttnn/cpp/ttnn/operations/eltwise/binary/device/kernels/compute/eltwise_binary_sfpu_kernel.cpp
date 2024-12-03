// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    unary_op_init_common(cb_in0, cb_out0);
    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_in0, per_core_block_size);
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_in0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in1, i, i * 2 + 1);
            eltwise_binop_tile_init();
#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();
        tile_regs_release();

        cb_pop_front(cb_in0, per_core_block_size);
        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
