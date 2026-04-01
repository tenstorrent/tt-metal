// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/sfpu_split_includes.h"
#include "api/compute/eltwise_unary/trigonometry.h"
#include "api/compute/mul_int_sfpu.h"
#include "api/compute/eltwise_unary/rpow.h"
#include "api/compute/eltwise_unary/rdiv.h"
#include "api/compute/eltwise_unary/fill.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    experimental::CircularBuffer cb_in(cb_input);
    experimental::CircularBuffer cb_out(cb_output);

    init_sfpu(cb_input, cb_output);
    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        copy_tile(cb_input, 0, 0);

#ifdef SFPU_OP_CHAIN_0
        SFPU_OP_CHAIN_0
#endif

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);

        cb_in.pop_front(1);
        cb_out.push_back(1);

        tile_regs_release();
    }
}
