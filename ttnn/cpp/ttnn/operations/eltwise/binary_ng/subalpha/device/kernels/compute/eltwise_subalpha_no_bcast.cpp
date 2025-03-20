// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/negative.h"

namespace NAMESPACE {
void MAIN {

    using namespace ckernel;
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input_a = tt::CBIndex::c_0;
    constexpr auto cb_input_b = tt::CBIndex::c_1;
    constexpr auto cb_out = tt::CBIndex::c_2;       // cb for output
    constexpr auto cb_alpha = tt::CBIndex::c_3;     // cb for alpha
    constexpr auto cb_inter = tt::CBIndex::c_4;     // intermediate cb

    constexpr uint32_t onetile = 1;
    binary_op_init_common(cb_input_a, cb_inter, cb_out);

    // wait input cb_alpha tile
    cb_wait_front(cb_alpha, onetile);

    for (uint32_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
        // wait input cb_input_b tile
        cb_wait_front(cb_input_b, onetile);

        // reserve cb_inter tile
        cb_reserve_back(cb_inter, onetile);

        mul_tiles_init(cb_input_b, cb_alpha);

        tile_regs_acquire();
        mul_tiles(cb_input_b, cb_alpha, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_inter);
        tile_regs_release();

        cb_pop_front(cb_input_b, onetile);
        cb_push_back(cb_inter, onetile);

        cb_wait_front(cb_input_a, onetile);

        cb_wait_front(cb_inter, onetile);
        cb_reserve_back(cb_out, onetile);

        sub_tiles_init(cb_input_a, cb_inter);
        tile_regs_acquire();
        sub_tiles(cb_input_a, cb_inter, 0, 0, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_out);
        tile_regs_release();

        cb_push_back(cb_out, onetile);
        cb_pop_front(cb_inter, onetile);
        cb_pop_front(cb_input_a, onetile);
    }
    cb_pop_front(cb_alpha, onetile);
}
}  // namespace NAMESPACE
