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
void MAIN {

    using namespace ckernel;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input_a = tt::CBIndex::c_0;
    constexpr auto cb_input_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;

    constexpr auto cb_a_inter = tt::CBIndex::c_3;  // intermediate cb for a^2
    constexpr auto cb_b_inter = tt::CBIndex::c_4;  // intermediate cb for b^2

    constexpr uint32_t onetile = 1;
    binary_op_init_common(cb_a_inter, cb_b_inter, cb_out);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {

        // for a^2
        reconfig_data_format_srca(cb_a_inter, cb_input_a);
        pack_reconfig_data_format(cb_out, cb_a_inter);

        cb_wait_front(cb_input_a, onetile);
        cb_reserve_back(cb_a_inter, onetile);

        tile_regs_acquire();
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile_to_dst_init_short(cb_input_a);
            copy_tile(cb_input_a, i, i);  // cb to DST
            square_tile_init();
            square_tile(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < onetile; ++i) {
            pack_tile(i, cb_a_inter);
        }
        tile_regs_release();

        cb_pop_front(cb_input_a, onetile);
        cb_push_back(cb_a_inter, onetile);

        reconfig_data_format_srca(/*old*/ cb_input_a, /*new*/ cb_a_inter);
        pack_reconfig_data_format(/*old*/ cb_a_inter, /*new*/ cb_out);

        // wait input a^2 tile
        cb_wait_front(cb_a_inter, onetile);

        // for b^2
        reconfig_data_format_srca(cb_a_inter, cb_input_b);
        pack_reconfig_data_format(cb_out, cb_b_inter);

        cb_wait_front(cb_input_b, onetile);
        cb_reserve_back(cb_b_inter, onetile);

        tile_regs_acquire();
        for (uint32_t i = 0; i < onetile; ++i) {
            copy_tile_to_dst_init_short(cb_input_b);
            copy_tile(cb_input_b, i, i);
            square_tile_init();
            square_tile(i);
        }
        tile_regs_commit();

        tile_regs_wait();
        for (uint32_t i = 0; i < onetile; ++i) {
            pack_tile(i, cb_b_inter);
        }
        tile_regs_release();

        cb_pop_front(cb_input_b, onetile);
        cb_push_back(cb_b_inter, onetile);

        reconfig_data_format_srca(/*old*/ cb_input_b, /*new*/ cb_a_inter);
        pack_reconfig_data_format(/*old*/ cb_b_inter, /*new*/ cb_out);

        // wait input b^2 tile
        cb_wait_front(cb_b_inter, onetile);
        // reserve out tile
        cb_reserve_back(cb_out, onetile);

        add_tiles_init(cb_a_inter, cb_b_inter);

        tile_regs_acquire();
        add_tiles(cb_a_inter, cb_b_inter, 0, 0, 0);
        sqrt_tile_init();
        sqrt_tile(0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_a_inter, onetile);
        cb_pop_front(cb_b_inter, onetile);
    }
}
}  // namespace NAMESPACE
