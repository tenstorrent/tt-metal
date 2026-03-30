// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/logsigmoid.h"
#include "api/compute/eltwise_unary/negative.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    init_sfpu(cb_input, cb_output);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        cb_wait_front(cb_input, 1);
        cb_reserve_back(cb_output, 1);

        tile_regs_acquire();

        copy_tile_to_dst_init_short(cb_input);
        copy_tile(cb_input, 0, 0);
        copy_tile(cb_input, 0, 1);

        negative_tile_init();
        negative_tile(1);

        exp_tile_init<true, true>();
        exp_tile<true, true>(1);

        logsigmoid_tile_init();
        logsigmoid_tile(0, 1, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);
        tile_regs_release();

        cb_pop_front(cb_input, 1);
        cb_push_back(cb_output, 1);
    }
}
