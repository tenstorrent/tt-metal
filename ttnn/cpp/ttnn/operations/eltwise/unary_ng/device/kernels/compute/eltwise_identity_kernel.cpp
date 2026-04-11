// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "experimental/circular_buffer.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_input = tt::CBIndex::c_0;
    constexpr auto cb_output = tt::CBIndex::c_2;

    experimental::CircularBuffer cb_in(cb_input);
    experimental::CircularBuffer cb_out(cb_output);

    init_sfpu(cb_input, cb_output);
    copy_tile_init(cb_input);

    for (uint32_t i = 0; i < num_tiles; ++i) {
        tile_regs_acquire();

        cb_in.wait_front(1);
        cb_out.reserve_back(1);

        copy_tile(cb_input, 0, 0);

        tile_regs_commit();
        tile_regs_wait();

        pack_tile(0, cb_output);

        cb_in.pop_front(1);
        cb_out.push_back(1);

        tile_regs_release();
    }
}
