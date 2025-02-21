// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/sqrt.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"

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
    #define CB_PRE_BCAST cb_pre_rhs
    #define CB_POST_BCAST cb_post_rhs
    #define CB_PRE_OTHER cb_pre_lhs
    #define CB_POST_OTHER cb_post_lhs
    #else
    #define CB_PRE_BCAST cb_pre_lhs
    #define CB_POST_BCAST cb_post_lhs
    #define CB_PRE_OTHER cb_pre_rhs
    #define CB_POST_OTHER cb_post_rhs
    #endif

    reconfig_data_format_srca(CB_POST_BCAST, CB_PRE_BCAST);
    pack_reconfig_data_format(cb_out, CB_POST_BCAST);

    cb_wait_front(CB_PRE_BCAST, onetile);
    cb_reserve_back(CB_POST_BCAST, onetile);

    tile_regs_acquire();
    for (uint32_t i = 0; i < onetile; ++i) {
        copy_tile_to_dst_init_short(CB_PRE_BCAST);
        copy_tile(CB_PRE_BCAST, i, i);
        square_tile_init();
        square_tile(i);
    }
    tile_regs_commit();

    tile_regs_wait();
    for (uint32_t i = 0; i < onetile; ++i) {
        pack_tile(i, CB_POST_BCAST);
    }
    tile_regs_release();

    cb_pop_front(CB_PRE_BCAST, onetile);
    cb_push_back(CB_POST_BCAST, onetile);

    reconfig_data_format_srca(CB_PRE_BCAST, CB_POST_BCAST);
    pack_reconfig_data_format(CB_POST_BCAST, cb_out);

    cb_wait_front(CB_POST_BCAST, onetile);

    // Compute on other cb (no-bcast reqd)
    for (uint32_t j = tile_start; j < freq; ++j) {
        reconfig_data_format_srca(CB_POST_OTHER, CB_PRE_OTHER);
        pack_reconfig_data_format(cb_out, CB_POST_OTHER);

        cb_wait_front(CB_PRE_OTHER, onetile);
        cb_reserve_back(CB_POST_OTHER, onetile);

        tile_regs_acquire();
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile_to_dst_init_short(CB_PRE_OTHER);
            copy_tile(CB_PRE_OTHER, i, i);
            square_tile_init();
            square_tile(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < onetile; ++i) {
            pack_tile(i, CB_POST_OTHER);
        }
        tile_regs_release();

        cb_pop_front(CB_PRE_OTHER, onetile);
        cb_push_back(CB_POST_OTHER, onetile);

        reconfig_data_format_srca(CB_PRE_OTHER, CB_POST_OTHER);
        pack_reconfig_data_format(CB_POST_OTHER, cb_out);

        cb_wait_front(CB_POST_OTHER, onetile);
        cb_reserve_back(cb_out, onetile);

        tile_regs_acquire();
        add_tiles_init(cb_post_lhs, cb_post_rhs);
        add_tiles(cb_post_lhs, cb_post_rhs, 0, 0, 0);
        sqrt_tile_init();
        sqrt_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(CB_POST_OTHER, onetile);
    }
    cb_pop_front(CB_POST_BCAST, onetile);
}

void MAIN {

    using namespace ckernel;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t tile_freq = get_arg_val<uint32_t>(1);
    uint32_t tile_start = get_arg_val<uint32_t>(2);

    if (num_tiles == 0) {
        return;
    }

    constexpr auto cb_input_a = tt::CBIndex::c_0;
    constexpr auto cb_input_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_a_inter = tt::CBIndex::c_3;  // intermediate cb for a^2
    constexpr auto cb_b_inter = tt::CBIndex::c_4;  // intermediate cb for b^2

    constexpr uint32_t onetile = 1;
    binary_op_init_common(cb_a_inter, cb_b_inter, cb_out);

    uint32_t complete_iterations = (num_tiles + tile_start) / tile_freq;
    uint32_t remaining_iterations = (num_tiles + tile_start) % tile_freq;

    for (uint32_t i = 0; i < complete_iterations; ++i, tile_start = 0) {
        process_tile(cb_input_a, cb_a_inter, cb_input_b, cb_b_inter, cb_out, tile_freq, tile_start);
    }

    if (remaining_iterations > 0) {
        process_tile(cb_input_a, cb_a_inter, cb_input_b, cb_b_inter, cb_out, remaining_iterations, tile_start);
    }
}
}// namespace NAMESPACE
