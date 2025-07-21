// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#define PRE_SCALE defined SFPU_OP_INIT_PRE_IN0_0 || defined SFPU_OP_INIT_PRE_IN1_0

namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;

#ifdef SFPU_OP_INIT_PRE_IN0_0
    constexpr auto cb_inp0 = tt::CBIndex::c_3;
#else
    constexpr auto cb_inp0 = cb_in0;
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
    constexpr auto cb_inp1 = tt::CBIndex::c_4;
#else
    constexpr auto cb_inp1 = cb_in1;
#endif
    constexpr auto cb_out0 = tt::CBIndex::c_2;

    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

#if not PRE_SCALE
    binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_inp0, cb_inp1);
#endif

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {
#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
        reconfig_data_format_srca(cb_inp0, cb_in0);
        pack_reconfig_data_format(cb_out0, cb_inp0);
        cb_wait_front(cb_in0, per_core_block_size);
        cb_reserve_back(cb_inp0, per_core_block_size);

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN0_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in0, i, i);  // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN0_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp0);  // DST[0]->cb
        }
        tile_regs_release();

        cb_pop_front(cb_in0, per_core_block_size);
        cb_push_back(cb_inp0, per_core_block_size);
#ifndef SFPU_OP_INIT_PRE_IN1_0
        reconfig_data_format_srca(cb_in0, cb_inp0);
        pack_reconfig_data_format(cb_inp0, cb_out0);
#endif
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
#ifndef SFPU_OP_INIT_PRE_IN0_0
        reconfig_data_format_srca(cb_inp0, cb_in1);
        pack_reconfig_data_format(cb_out0, cb_inp1);
#else
        reconfig_data_format_srca(cb_in0, cb_in1);
        pack_reconfig_data_format(cb_inp0, cb_inp1);
#endif
        cb_wait_front(cb_in1, per_core_block_size);
        cb_reserve_back(cb_inp1, per_core_block_size);

        tile_regs_acquire();
        SFPU_OP_INIT_PRE_IN1_0
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_in1, i, i);  // copy from c_in[0] to DST[0]
            SFPU_OP_FUNC_PRE_IN1_0
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_inp1);  // DST[0]->cb
        }
        tile_regs_release();

        cb_pop_front(cb_in1, per_core_block_size);
        cb_push_back(cb_inp1, per_core_block_size);
        reconfig_data_format_srca(cb_in1, cb_inp0);
        pack_reconfig_data_format(cb_inp1, cb_out0);
#endif

        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

#if PRE_SCALE
        binary_tiles_init<true, ELTWISE_OP_TYPE>(cb_inp0, cb_inp1);
#endif

        tile_regs_acquire();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            ELTWISE_OP(cb_inp0, cb_inp1, i, i, i);

#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);
        }
        tile_regs_release();

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
