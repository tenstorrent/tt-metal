// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api.h"

namespace NAMESPACE {

//
// Compute kernel for elemwise y = ax + b
// Assumptions
// - a is stored in c_0, x in c_1, b in c_2
//
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);

    // input
    constexpr auto cb_id_a = tt::CBIndex::c_0;
    constexpr auto cb_id_x = tt::CBIndex::c_1;
    constexpr auto cb_id_b = tt::CBIndex::c_2;

    // output
    constexpr auto cb_id_y = tt::CBIndex::c_16;

    constexpr uint32_t dst_reg_id = 0;

    binary_op_init_common(cb_id_a, cb_id_x, cb_id_y);

    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_wait_front(cb_id_b, 1);

        tile_regs_acquire();

        // b->dest
        copy_tile_to_dst_init_short(cb_id_b);
        copy_tile(cb_id_b, 0, dst_reg_id);
        cb_pop_front(cb_id_b, 1);

        // ax+dest->dest
        cb_wait_front(cb_id_a, 1);
        cb_wait_front(cb_id_x, 1);
        mul_tiles_init(cb_id_a, cb_id_x);
        mul_tiles(cb_id_a, cb_id_x, 0, 0, dst_reg_id);

        tile_regs_commit();
        tile_regs_wait();

        // y -> reader
        cb_reserve_back(cb_id_y, 1);
        pack_tile(dst_reg_id, cb_id_y);
        cb_push_back(cb_id_y, 1);

        cb_pop_front(cb_id_a, 1);
        cb_pop_front(cb_id_x, 1);

        // Release the held register
        tile_regs_release();
    }
}
}  // namespace NAMESPACE
