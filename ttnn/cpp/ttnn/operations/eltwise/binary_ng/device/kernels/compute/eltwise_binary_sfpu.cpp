// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
// #include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"

#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api/add_int32_sfpu.h"
#include "eltwise_defines.hpp"
#include "eltwise_utils.hpp"

namespace NAMESPACE {

ALWI void process_tile(
    tt::CBIndex cb_pre_lhs,
    tt::CBIndex cb_post_lhs,
    tt::CBIndex cb_pre_rhs,
    tt::CBIndex cb_post_rhs,
    tt::CBIndex cb_out,
    uint32_t freq,
    uint32_t tile_start) {
    using namespace ckernel;
    constexpr uint32_t onetile = 1;

#if BCAST_INPUT
    auto cb_bcast = cb_post_rhs;
    auto cb_other = cb_post_lhs;
#else
    auto cb_bcast = cb_post_lhs;
    auto cb_other = cb_post_rhs;
#endif

#if SFPU_OP_INIT_PRE_IN0_0 && (BCAST_INPUT == 0)
    PREPROCESS_SFPU_A(cb_pre_lhs, cb_post_lhs, cb_out, onetile);
#elif SFPU_OP_INIT_PRE_IN1_0 && (BCAST_INPUT == 1)
    PREPROCESS_SFPU_B(cb_pre_rhs, cb_post_rhs, cb_out, onetile);
#endif

    cb_wait_front(cb_bcast, onetile);

    for (uint32_t j = tile_start; j < freq; ++j) {
#if SFPU_OP_INIT_PRE_IN0_0 && (BCAST_INPUT == 1)
        PREPROCESS_SFPU_A(PREPROCESS_A_INIT, PREPROCESS_A_APPLY, cb_pre_lhs, cb_post_lhs, cb_out, onetile);
#elif SFPU_OP_INIT_PRE_IN1_0 && (BCAST_INPUT == 0)
        PREPROCESS_SFPU_B(PREPROCESS_B_INIT, PREPROCESS_B_APPLY, cb_pre_rhs, cb_post_rhs, cb_out, onetile);
#endif
        cb_wait_front(cb_other, onetile);

        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short_with_dt(cb_post_rhs, cb_post_lhs);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_post_lhs, i, i * 2);
        }
        copy_tile_to_dst_init_short_with_dt(cb_post_lhs, cb_post_rhs);
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile(cb_post_rhs, i, i * 2 + 1);

            // BINARY_OP(cb_post_lhs, cb_post_rhs, 0, 0, 0);
#ifndef INT32_INIT
            eltwise_binop_tile_init();
#endif

#ifdef ADD_INT32_INIT
            ADD_INT32_INIT
#endif
#ifdef BITWISE_INIT
            BITWISE_INIT
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
            tile_regs_commit();
            pack_tile(i * 2, cb_out);
        }
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_other, onetile);
    }
    cb_pop_front(cb_bcast, onetile);
}

void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_pre_lhs = tt::CBIndex::c_0;
    constexpr auto cb_pre_rhs = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_post_lhs = PREPROCESS_A ? tt::CBIndex::c_3 : cb_pre_lhs;
    constexpr auto cb_post_rhs = PREPROCESS_B ? tt::CBIndex::c_4 : cb_pre_rhs;

    // binary_op_init_common(cb_post_lhs, cb_post_rhs, cb_out);
    unary_op_init_common(cb_post_lhs, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out, tile_freq, tile_start);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_pre_lhs, cb_post_lhs, cb_pre_rhs, cb_post_rhs, cb_out, remaining_iterations, tile_start);
    }
}
}  // namespace NAMESPACE
