// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api/add_int_sfpu.h"
#include "compute_kernel_api/sub_int_sfpu.h"
#include "compute_kernel_api/mul_int_sfpu.h"
#include "compute_kernel_api/binary_max_min.h"
#include "compute_kernel_api/gcd.h"
#include "compute_kernel_api/lcm.h"

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

    unary_op_init_common(cb_in0, cb_out0);

#ifdef PACK_RELU
    PACK((llk_pack_relu_config(ReluType::ZERO_RELU)));
#endif

    for (uint32_t block = 0; block < per_core_block_cnt; ++block) {

#if PRE_SCALE
        copy_tile_to_dst_init_short(cb_in0);  // need to copy from CB to DST to be able to run sfpu math
#endif

#ifdef SFPU_OP_INIT_PRE_IN0_0
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
#endif

#ifdef SFPU_OP_INIT_PRE_IN1_0
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
#endif
        cb_wait_front(cb_inp0, per_core_block_size);
        cb_wait_front(cb_inp1, per_core_block_size);
        cb_reserve_back(cb_out0, per_core_block_size);

        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short_with_dt(cb_inp1, cb_inp0);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp0, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_inp0, cb_inp1);
        for (uint32_t i = 0; i < per_core_block_size; ++i) {
            copy_tile(cb_inp1, i, i * 2 + 1);

#ifdef BINOP_INIT
            BINOP_INIT
#endif
#ifdef ADD_INT_INIT
            ADD_INT_INIT
#endif
#ifdef SUB_INT_INIT
            SUB_INT_INIT
#endif
#ifdef MUL_INT_INIT
            MUL_INT_INIT
#endif
#ifdef BITWISE_INIT
            BITWISE_INIT
#endif
#ifdef BITWISE_UINT16_INIT
            BITWISE_UINT16_INIT
#endif
#ifdef SHIFT_INIT
            SHIFT_INIT
#endif

#ifdef BINARY_SFPU_OP
            BINARY_SFPU_OP
#endif
#ifdef SFPU_OP_INIT_0
            SFPU_OP_INIT_0
            SFPU_OP_FUNC_0
#endif

#ifdef SFPU_OP_CHAIN_0
            SFPU_OP_CHAIN_0
#endif
            pack_tile(i * 2, cb_out0);
        }
        tile_regs_commit();
        tile_regs_release();

        cb_pop_front(cb_inp0, per_core_block_size);
        cb_pop_front(cb_inp1, per_core_block_size);
        cb_push_back(cb_out0, per_core_block_size);
    }
}
}  // namespace NAMESPACE
